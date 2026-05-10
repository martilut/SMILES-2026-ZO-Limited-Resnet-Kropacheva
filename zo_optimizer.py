"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Callable

import torch
import torch.nn as nn

from config import OPTIMIZATION_MODE, N_BATCHES, SPSA_K, FREEZE


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.modes = self._init_optimization_modes()
        self.save_mode_param_counts()
        self.mode = OPTIMIZATION_MODE
        self.steps = 0
        if self.mode in ("dynamic", "dynamic_reverse"):
            self.layer_names: list[str] = self.modes["fc"]
        else:
            self.layer_names: list[str] = self.modes[self.mode]
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _init_optimization_modes(self) -> dict[str, list[str]]:
        all_params = [n for n, _ in self.model.named_parameters()]
        fc = [n for n in all_params if n.startswith("fc.")]

        def is_bn(n: str) -> bool:
            return (
                    n.startswith("bn1.")
                    or ".bn1." in n or ".bn2." in n
                    or ".downsample.1." in n
            )

        bn_all = [n for n in all_params if is_bn(n)]

        bn_stem = [n for n in bn_all if n.startswith("bn1.")]
        bn_block = [n for n in bn_all if (".bn1." in n or ".bn2." in n)]
        bn_down = [n for n in bn_all if ".downsample.1." in n]

        def is_conv(n: str) -> bool:
            return (
                    n == "conv1.weight"
                    or ".conv1.weight" in n or ".conv2.weight" in n
                    or ".downsample.0.weight" in n
            )

        conv_all = [n for n in all_params if is_conv(n)]

        last_conv = ["layer4.1.conv2.weight"]
        layer4_convs = [n for n in conv_all if n.startswith("layer4.")]
        layer4_1 = [n for n in all_params if n.startswith("layer4.1.")]
        layer4_0 = [n for n in all_params if n.startswith("layer4.0.")]

        modes = {
            "fc": fc,
            "fc_bn_stem": fc + bn_stem,
            "fc_bn_block": fc + bn_block,
            "fc_bn_down": fc + bn_down,
            "fc_bn_all": fc + bn_all,
            "fc_last_conv": fc + last_conv,
            "fc_bn_all_last_conv": fc + bn_all + last_conv,
            "fc_bn_all_layer4": fc + bn_all + layer4_convs,
            "bn_all": bn_all,
            "last_conv_only": last_conv,
            "layer4_1": layer4_1,
            "layer4_0": layer4_0,
        }
        return modes

    def _update_dynamic_schedule(self) -> None:
        progress = self.steps / N_BATCHES
        if progress < 1 / 3:
            self.layer_names = self.modes["fc"]
        elif progress < 2 / 3:
            self.layer_names = self.modes["bn_all"] \
                if FREEZE else self.modes["fc_bn_all"]
        else:
            self.layer_names = self.modes["last_conv_only"] \
                if FREEZE else self.modes["fc_bn_all_last_conv"]

    def _update_dynamic_reverse_schedule(self) -> None:
        progress = self.steps / N_BATCHES
        if progress < 1 / 3:
            self.layer_names = self.modes["fc"]
        elif progress < 2 / 3:
            self.layer_names = self.modes["layer4_1"] \
                if FREEZE else self.modes["fc"] + self.modes["layer4_1"]
        else:
            self.layer_names = self.modes["layer4_0"] \
                if FREEZE else self.modes["fc"] + self.modes["layer4_1"] + self.modes["layer4_0"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random unit-norm perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param``, normalised to unit L2 norm.
        """
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        norm = u.norm()
        if norm > 0:
            u = u / norm
        return u

    def _estimate_grad_spsa(
            self,
            loss_fn: Callable[[], float],
            params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        directions: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            directions[name] = self._sample_direction(param)

        with torch.no_grad():
            for name, param in params.items():
                param.data.add_(self.eps * directions[name])
            f_plus = loss_fn()

            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * directions[name])
            f_minus = loss_fn()

            for name, param in params.items():
                param.data.add_(self.eps * directions[name])

        scalar_grad = (f_plus - f_minus) / (2.0 * self.eps)
        return {n: scalar_grad * directions[n] for n in params}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Skeleton: 2-point central-difference estimator.
        For each active parameter ``p`` independently:
            1. Sample a random unit vector ``u`` of the same shape as ``p``.
            2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
            3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
            4. Restore ``p`` to its original value.
            5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

        This is an unbiased estimator of the directional derivative along ``u``
        scaled back to parameter space.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).

        Student task:
            Replace this with a more efficient or accurate estimator:
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------
        grads = {n: torch.zeros_like(p) for n, p in params.items()}
        for _ in range(SPSA_K):
            single_direction = self._estimate_grad_spsa(loss_fn, params)
            for n in grads:
                grads[n].add_(single_direction[n])
        return {n: g / SPSA_K for n, g in grads.items()}
        # ------------------------------------------------------------------

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(self.lr * grads[name])
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        if self.mode == "dynamic":
            self._update_dynamic_schedule()
        elif self.mode == "dynamic_reverse":
            self._update_dynamic_reverse_schedule()
        self.steps += 1

        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)

    def save_mode_param_counts(self) -> None:
        output_path = f"mode_param_counts.csv"
        if os.path.exists(output_path):
            return
        named = dict(self.model.named_parameters())
        total_model_params = sum(p.numel() for p in self.model.parameters())

        rows = []
        for mode_name, layer_names in self.modes.items():
            n_tensors = len(layer_names)
            n_params = sum(named[n].numel() for n in layer_names)
            rows.append({
                "mode": mode_name,
                "n_tensors": n_tensors,
                "n_params": n_params,
                "percent": round(100.0 * n_params / total_model_params, 2),
            })

        rows.append({
            "mode": "full",
            "n_tensors": len(named),
            "n_params": total_model_params,
            "percent": 100.0,
        })

        rows.sort(key=lambda r: r["n_params"])

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "n_tensors", "n_params", "percent"])
            writer.writeheader()
            writer.writerows(rows)

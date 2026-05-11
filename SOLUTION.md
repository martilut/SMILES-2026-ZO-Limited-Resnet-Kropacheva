# Zero-Order Fine-Tuning of ResNet18 on CIFAR100

---

## 1. Reproducibility

### Environment

- Python 3.10+
- PyTorch 2.x with CUDA support (tested on GTX 1650, 4 GB VRAM, CUDA 12.4)
- All other dependencies are listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/martilut/SMILES-2026-ZO-Limited-Resnet-Kropacheva.git
cd SMILES-2026-ZO-Limited-Resnet-Kropacheva
pip install -r requirements.txt
```

### Reproducing `results.json`

Run the official evaluation script with the configuration used for the final submission:

```bash
python -u validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json \
    --seed 42
```

### Implementation details required for reproducibility

- **Seeding.** `validate.py` calls `seed_everything(42)` and uses a fixed `torch.Generator` for the train dataloader. All randomness in my added code (perturbation sampling in the optimizer, subset selection in the prototype init) consumes the same global RNG state, so results are bit-identical across runs on the same hardware.
- **Determinism note.** `torch.use_deterministic_algorithms(True, warn_only=True)` is set in `validate.py`. Numerical differences may appear across CUDA vs. CPU runs (different reduction kernels), but the validation accuracy should match within ±0.5% as specified in the task.
- **Cache invalidation.** If `head_init.py` or `augmentation.py` (its validation transform branch) is modified, delete `prior_init.pt` to force the prototype-init to recompute. The cache only stores the final weight/bias tensors and does not validate that the inputs are unchanged.

### File modifications

| File                          | Modified |
|-------------------------------|----------|
| `zo_optimizer.py`             | Yes      |
| `head_init.py`                | Yes      |
| `augmentation.py`             | No       |
| `train_data.py`               | Yes      |
| `config.py`, `utils.py` (new) | Yes      |
| `model.py`, `validate.py`     | No       |

---

## 2. Final Solution

### 2.1 Final configuration

The final submitted configuration is:

| Parameter | Value                                     | Where |
|---|-------------------------------------------|---|
| `HEAD_INIT` | `prior`                                   | `config.py` |
| `OPTIMIZATION_MODE` | `fc_bn_all`                               | `config.py` |
| `N_BATCHES` | `64`                                      | CLI |
| `BATCH_SIZE` | `128`                                     | CLI |
| `SPSA_K` | `8`                                       | `config.py` |
| `MOVING_AVERAGE_COEFF` | `0.5` (momentum β)                        | `config.py` |
| `UPDATE_RULE` | `momentum`                                | `config.py` |
| `lr` | `1e-3`                                    | `config.py` |
| `eps` | `1e-3`                                    | `config.py` |
| `perturbation_mode` | `gaussian`                                | `config.py` |
| Augmentation | Resize + horizontal flip only             | `augmentation.py` |
| Train data | Full CIFAR100 train split (50 000 images) | `train_data.py` |

### 2.2 Modified components

#### Head initialization (`head_init.py`)

I replaced the skeleton's Kaiming initialization with a data-aware prototype initialization computed from the frozen pretrained backbone. The pipeline:

1. Load CIFAR100's training split with the validation transform pipeline (so the features match what `validate.py` will see at evaluation time).
2. Pass the first 100 images of each class through a fresh ImageNet-pretrained ResNet18 with `fc` replaced by `nn.Identity`, producing 100 × 512 feature vectors per class.
3. Compute the class-mean feature vector `μ_c ∈ ℝ^512` for each of the 100 classes.
4. Set the head weights to unit-normalized class means and zero bias:

   ```
   weight[c] = μ_c / ‖μ_c‖
   bias[c]   = 0
   ```

Because pretrained ImageNet features already cluster CIFAR100 images by class very well, this single initialization step achieves **53.06% top-1 validation accuracy with zero optimizer steps**. The computation is done outside the optimizer budget (it does not consume any of the 8192 samples), and the result is cached to `prior_init.pt` for instant reuse across runs.

#### Zero-order optimizer (`zo_optimizer.py`)

I replaced the per-tensor central-difference skeleton with a **Simultaneous Perturbation Stochastic Approximation (SPSA)** estimator combined with heavy-ball momentum:

- **SPSA estimator.** Each step samples one Gaussian `N(0, I)` direction per parameter tensor, scaled to unit L2 norm, and evaluates `f(θ + εu)` and `f(θ - εu)` to form the pseudo-gradient `(f₊ - f₋) / (2ε) · u`. Cost: 2 forward passes per step regardless of the number of active parameters.
- **Variance reduction via K samples.** The pseudo-gradient is averaged over `SPSA_K` independent direction samples to reduce per-step variance.
- **Heavy-ball momentum update.** `v ← β · v + g; θ ← θ - lr · v`. The momentum buffer is keyed by parameter name and lazily initialized, so it works transparently with dynamic layer schedules.
- **Layer-selection modes.** I implemented multiple named modes (`fc`, `fc_bn_all`, `fc_bn_all_last_conv`, several dynamic curricula) so different parameter subsets could be A/B tested without code changes.

#### Augmentation (`augmentation.py`)

Additional augmentation beyond the skeleton's resize + horizontal flip hurt validation accuracy. The final pipeline therefore retains the skeleton's minimal augmentation.

#### Train data (`train_data.py`)

I tested both the full CIFAR100 training set and a class-stratified subset. The full dataset gave lower variance across runs and was kept for the final pipeline.

### 2.3 What contributed most to the final metric

In descending order of impact on the final `val_accuracy_top1_finetuned`:

1. Data-aware head initialization.
2. SPSA + momentum providing
3. Minimal augmentation.

**The prototype initialization is the most important**, and the zero-order fine-tuning almost doesn't add anything on top of it.

---

## 3. Experiments and Failed Attempts

This section documents alternatives I tested but did not include in the final solution. 
The following results are the best results obtained across different configurations with the given components fixed.


### 3.1 Head initialization

I tested five initialization strategies. The `init_last_layer` function in `head_init.py` dispatches based on the `HEAD_INIT` config value.

| Strategy | `imagenet_head` | `init_head` | `finetuned` |
|---|---|---|---|
| Kaiming uniform | 0.37% | 1.21% |  1.51% |
| Xavier uniform | 0.37% | 1.37% | 2.07% |
| Orthogonal | 0.37% | 1.13% | 0.97% |
| Small-scale random (±0.01) | 0.37% | 1.36% | 1.87% |
| CIFAR100 class prior | 0.37% | 53.06% | [TODO]% |

### 3.2 Update rule

| Rule                      | `finetuned` |
|---------------------------|-------------|
| Default (no momentum)       | 49.98%      |
| Momentum, β=0.9** | 53.06%      |
| Adam (β₁=0.9, β₂=0.999)   | 49.84%      |

### 3.3 Layer-selection mode

The `OPTIMIZATION_MODE` config selects which parameters the SPSA estimator perturbs each step.

| Mode | # params | Description                                         | `val_accuracy_top1_finetuned` |
|---|---|-----------------------------------------------------|-------------------------------|
| `fc` | 51 300 | Baseline: tune only the new head                    | 53.06%                        |
| `fc_bn_all` | 60 900 | Add all BN affine parameters (γ, β)                 | 53.21%                        |
| `fc_last_conv` | 2 410 596 | Add `layer4.1.conv2` weight                         | 53.06%                       |
| `fc_bn_all_last_conv` | 2 420 196 | Combine BN and last conv                            | 53.06%                       |
| Dynamic reverse (cumulative) | varies | Schedule: `fc → fc+layer4.1 → fc+layer4.1+layer4.0` | 53.05%                       |

**Discussion.** I considered three classes of parameters to tune beyond the head:

1. **BN affine parameters** (γ, β across all BatchNorm layers). These are very cheap (~9 600 parameters total) and well-placed: they appear at every layer of the network, allowing the entire backbone's activation distribution to be re-normalized for CIFAR100's statistics without touching any conv weights. In `eval()` mode (which `validate.py` enforces for every `loss_fn` call), the BN running stats are frozen at their ImageNet values, but the BN affine parameters are still active in the forward pass and remain trainable via parameter perturbation.

2. **Late convolutional layers** (`layer4.1.conv2`, optionally `layer4.0` and `layer4.1` blocks). The intuition from classical transfer learning is that late layers encode task-specific features more amenable to adaptation, while early layers encode generic edge/texture detectors that transfer well from ImageNet. The cost is that even a single layer4 conv tensor contains 2.36M parameters — 47× more than the entire head.

3. **Dynamic schedules.** Two curricula were tested:
   - **Forward dynamic:** `fc → fc + BN → fc + BN + last conv` — start narrow, progressively widen.
   - **Reverse-depth dynamic:** `fc → fc + layer4.1 → fc + layer4.1 + layer4.0` — start at the output and progressively unfreeze backward through the network.

   Both schedules can also run in a strict-freeze mode where prior phases' parameters are frozen rather than continued.

---

## Appendix A — Caching

- `prior_init.pt`: cached head weights/biases from the prototype init. Recomputed on first run or after deletion. ~200 KB.
- `mode_param_counts.csv`: tabulates parameter counts for each `OPTIMIZATION_MODE`. Useful for understanding the variance / step-cost trade-off across modes.

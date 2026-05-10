"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn

from config import HEAD_INIT
from utils import load_prior_init


def init_kaiming(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

def init_xavier(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)

def init_orthogonal(layer: nn.Linear) -> None:
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)

def init_small_random(layer: nn.Linear) -> None:
    nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
    nn.init.zeros_(layer.bias)

def init_prior(layer: nn.Linear) -> None:
    num_classes, in_features = layer.weight.shape
    data = load_prior_init(num_classes, in_features)

    with torch.no_grad():
        layer.weight.copy_(data["weight"])
        layer.bias.copy_(data["bias"])


HEAD_INIT_STRATEGIES = {
    "kaiming": init_kaiming,
    "xavier": init_xavier,
    "orthogonal": init_orthogonal,
    "small_random": init_small_random,
    "prior": init_prior,
}


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    HEAD_INIT_STRATEGIES[HEAD_INIT](layer)
    # -------------------------------------------------------------------------

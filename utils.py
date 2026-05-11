from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

from augmentation import get_transforms
from config import BATCH_SIZE


def compute_prior_init(num_classes: int, in_features: int) -> dict[str, torch.Tensor]:
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()

    dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=get_transforms(train=False),
    )

    targets = dataset.targets
    per_class_indices: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        if len(per_class_indices[label]) < 100:
            per_class_indices[label].append(idx)

    selected_indices = [i for c in range(num_classes) for i in per_class_indices[c]]
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    feature_sums = torch.zeros(num_classes, in_features)
    feature_counts = torch.zeros(num_classes)

    with torch.no_grad():
        for images, labels in loader:
            features = backbone(images)

            feature_sums.index_add_(0, labels, features)
            feature_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))

    class_means = feature_sums / feature_counts.unsqueeze(1)

    weight = class_means / class_means.norm(dim=1, keepdim=True)
    bias = torch.zeros(num_classes)

    return {
        "weight": weight.cpu(),
        "bias": bias.cpu(),
    }


def load_prior_init(num_classes: int, in_features: int) -> dict[str, torch.Tensor]:
    cache = Path("prior_init.pt")

    if cache.exists():
        print(f"Loading prior init from cache: {cache}")
        return torch.load(cache, map_location="cpu", weights_only=True)

    print(f"Computing prior init and caching to: {cache}")
    data = compute_prior_init(num_classes, in_features)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)

    return data

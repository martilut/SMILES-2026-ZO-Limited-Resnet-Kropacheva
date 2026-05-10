import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

from augmentation import get_transforms
from config import SUBSET_SIZE, SEED

USE_TRAIN_SUBSET_ONLY=True

def make_stratified_subset(dataset):
    labels = np.array(dataset.targets)
    num_classes = int(labels.max()) + 1

    class_indices = {cls: np.where(labels == cls)[0].tolist()
                     for cls in range(num_classes)}

    rng = np.random.default_rng(SEED)
    for cls in class_indices:
        rng.shuffle(class_indices[cls])

    per_class = SUBSET_SIZE // num_classes
    remainder = SUBSET_SIZE % num_classes

    selected = []
    for cls in range(num_classes):
        n = per_class + (1 if cls < remainder else 0)
        selected.extend(class_indices[cls][:n])

    return Subset(dataset, selected)


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )

    if SUBSET_SIZE is not None:
        train_dataset = make_stratified_subset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader

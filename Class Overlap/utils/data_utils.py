import numpy as np
import random
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


def construct_transform(rescale: int = None, normalizer=None):
    """
    Build a preprocessing transformation pipeline.

    Parameters
    ----------
    rescale : int, optional
        If provided, images are resized to this resolution.

    normalizer : torchvision transform, optional
        Normalization transform applied after ToTensor().

    Returns
    -------
    torchvision.transforms.Compose
        A composed transformation.
    """
    if rescale is not None:
        tfs = [transforms.Resize(rescale), transforms.ToTensor(), normalizer]
    else:
        tfs = [transforms.ToTensor(), normalizer]

    return transforms.Compose(tfs)


def load_cifar10(rescale: int = None, normalizer=None):
    """
    Load the CIFAR-10 dataset.

    Parameters
    ----------
    rescale : int, optional
        Passed to construct_transform.

    normalizer : optional
        Custom normalization transform.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale,
            normalizer=transforms.Normalize(
                (0.4913997, 0.48215827, 0.4465312),
                (0.2470323, 0.2434850, 0.2615877),
            ),
        )
    else:
        transform = construct_transform(rescale, normalizer)

    return (
        datasets.CIFAR10("data", train=True, download=True, transform=transform),
        datasets.CIFAR10("data", train=False, download=True, transform=transform),
    )


def load_cifar100(rescale: int = None, normalizer=None):
    """
    Load the CIFAR-100 dataset.

    Parameters
    ----------
    rescale : int, optional
        Passed to construct_transform.

    normalizer : optional
        Custom normalization transform.
    """
    if normalizer is None:
        transform = construct_transform(
            rescale,
            normalizer=transforms.Normalize(
                (0.5070746, 0.4865490, 0.4409179),
                (0.2673342, 0.2564385, 0.2761506),
            ),
        )
    else:
        transform = construct_transform(rescale, normalizer)

    return (
        datasets.CIFAR100("data", train=True, download=True, transform=transform),
        datasets.CIFAR100("data", train=False, download=True, transform=transform),
    )


class LabelNoiseDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that injects random label noise.

    It simulates class confusion by randomly replacing
    a proportion of labels with labels from other classes.

    Example:
    25% of class A may be relabeled as class B.
    """

    def __init__(self, dataset, n_classes, noise_prop=0) -> None:
        super().__init__()

        self.dataset = dataset
        self.new_labels = [0] * len(dataset)

        # Create a random mapping from each class
        # to a different class
        self.class_mapping = {
            i: random.choice(list(range(i)) + list(range(i + 1, n_classes)))
            for i in range(n_classes)
        }

        # Apply noise according to noise_prop
        for i in range(len(dataset)):
            _, y = dataset[i]

            if np.random.rand() < noise_prop:
                self.new_labels[i] = self.class_mapping[y]
            else:
                self.new_labels[i] = y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x, self.new_labels[index]


def load_dataset(
    dataset: str,
    rescale: int = None,
    custom_normalizer=None,
    subsample: int = 0,
    label_noise: float = 0,
):
    """
    Load a dataset specified by name.

    Parameters
    ----------
    dataset : str
        Dataset identifier (e.g., CIFAR10, CIFAR100, MNIST, etc.).

    rescale : int, optional
        Image resizing parameter.

    custom_normalizer : optional
        Custom normalization transform.

    subsample : int, optional
        Number of samples to keep (random subset).
        0 means no subsampling.

    label_noise : float, optional
        Proportion of labels to corrupt.
    """

    out_dim = 10
    n_channels = 3  # Default number of input channels

    if dataset == "CIFAR10":
        train_data, test_data = load_cifar10(rescale, custom_normalizer)

    elif dataset == "CIFAR100":
        out_dim = 100
        train_data, test_data = load_cifar100(rescale, custom_normalizer)

        mus = torch.FloatTensor([0, label_noise])
        out_dim = 2

    else:
        sys.exit(f"Dataset {dataset} is an invalid dataset.")

    # -----------------------
    # Optional Subsampling
    # -----------------------
    if subsample > 0:
        train_data = torch.utils.data.Subset(
            train_data,
            np.random.choice(
                list(range(len(train_data))),
                size=subsample,
                replace=False,
            ),
        )

        test_data = torch.utils.data.Subset(
            test_data,
            np.random.choice(
                list(range(len(test_data))),
                size=int(0.2 * subsample),
                replace=False,
            ),
        )

    # -----------------------
    # Optional Label Noise
    # -----------------------
    if label_noise > 0 and dataset != "Gaussians":
        train_data = LabelNoiseDataset(
            dataset=train_data,
            n_classes=out_dim,
            noise_prop=label_noise,
        )

    return train_data, test_data, n_channels, out_dim


def split_train_into_val(train_data, val_prop: float = 0.1):
    """
    Split a training dataset into training and validation subsets.

    Parameters
    ----------
    train_data : Dataset
        Full training dataset.

    val_prop : float
        Proportion of data allocated to validation.

    Returns
    -------
    train_subset, val_subset
    """

    val_len = int(val_prop * len(train_data))

    train_subset, val_subset = torch.utils.data.random_split(
        train_data,
        [len(train_data) - val_len, val_len],
    )

    return train_subset, val_subset

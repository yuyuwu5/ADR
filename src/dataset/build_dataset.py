import os

import torch
import torchvision
import torchvision.transforms as T

from config import CIFAR10_DIR, CIFAR100_DIR, TINY_IMAGENET_DIR
from dataset.aux_dataset import AuxDDPM


def build_dataset(dataset_name, is_train=False, aux_dataset_path=None):
    if dataset_name == "tiny-imagenet":
        image_size = 64
    else:
        image_size = 32

    if is_train:
        transform = T.Compose(
            [
                T.RandomCrop(image_size, padding=4),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    if aux_dataset_path is not None:
        dataset = AuxDDPM(aux_dataset_path, transform=transform)
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=CIFAR10_DIR, train=is_train, download=False, transform=transform
        )
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=CIFAR100_DIR, train=is_train, download=False, transform=transform
        )
    elif dataset_name == "tiny-imagenet":
        dataset_path = (
            os.path.join(TINY_IMAGENET_DIR, "train")
            if is_train
            else os.path.join(TINY_IMAGENET_DIR, "val")
        )
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    else:
        raise NotImplementedError("No such dataset: %s" % (dataset_name))
    return dataset

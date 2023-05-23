import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CIFAR10_DIR = os.path.join(DATA_DIR, "cifar10")
CIFAR100_DIR = os.path.join(DATA_DIR, "cifar100")
TINY_IMAGENET_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")

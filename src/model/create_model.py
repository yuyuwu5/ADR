import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.preact_resnet import PreActResNet
from model.resnet import ResNet
from model.wideresnet import WideResNet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)


class _Swish(torch.autograd.Function):
    """Custom implementation of swish."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Module using custom implementation."""

    def forward(self, input_tensor):
        return _Swish.apply(input_tensor)


@gin.configurable
def create_model(model_name, dataset_name="cifar10", activation_name="relu"):
    if activation_name == "relu":
        activation = nn.ReLU
    elif activation_name == "swish":
        activation = Swish
    else:
        raise NotImplementedError("Not support such activation: %s" % (activation_name))

    if dataset_name == "cifar10":
        num_classes = 10
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
        adaptive_pooling = False
    elif dataset_name == "cifar100":
        num_classes = 100
        mean = CIFAR100_MEAN
        std = CIFAR100_STD
        adaptive_pooling = False
    elif dataset_name == "tiny-imagenet":
        num_classes = 200
        mean = TINY_IMAGENET_MEAN
        std = TINY_IMAGENET_STD
        adaptive_pooling = True
    else:
        raise NotImplementedError("Not support dataset: %s" % (dataset_name))

    if model_name == "preact-resnet18":
        model = PreActResNet(
            mean,
            std,
            num_classes=num_classes,
            depth=18,
            activation_fn=activation,
            adaptive_pooling=adaptive_pooling,
        )
    elif model_name == "resnet18":
        model = ResNet(
            mean,
            std,
            num_classes=num_classes,
            depth=18,
            activation_fn=activation,
            adaptive_pooling=adaptive_pooling,
        )
    elif model_name == "wideresnet-34-10":
        model = WideResNet(
            mean,
            std,
            num_classes=num_classes,
            depth=34,
            width=10,
            activation_fn=activation,
            adaptive_pooling=adaptive_pooling,
        )
    else:
        raise NotImplementedError("Not support such model: %s" % (model_name))
    return model

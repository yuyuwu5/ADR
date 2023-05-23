from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PreActBlock(nn.Module):
    """Pre-activation ResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self._stride = stride
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        # We manually pad to obtain the same effect as `SAME` (necessary when
        # `stride` is different than 1).
        self.conv_2d_1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_2d_2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.has_shortcut = stride != 1 or in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=0,
                bias=False,
            )

    def _pad(self, x):
        if self._stride == 1:
            x = F.pad(x, (1, 1, 1, 1))
        elif self._stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride`.")
        return x

    def forward(self, x):
        out = self.relu_0(self.batchnorm_0(x))
        shortcut = self.shortcut(self._pad(x)) if self.has_shortcut else x
        out = self.conv_2d_1(self._pad(out))
        out = self.conv_2d_2(self.relu_1(self.batchnorm_1(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    """Pre-activation ResNet."""

    def __init__(
        self,
        mean: Union[Tuple[float, ...], float],
        std: Union[Tuple[float, ...], float],
        num_classes: int = 10,
        depth: int = 18,
        width: int = 0,  # Used to make the constructor consistent.
        activation_fn: nn.Module = nn.ReLU,
        padding: int = 0,
        num_input_channels: int = 3,
        adaptive_pooling=False,
    ):
        super().__init__()
        if width != 0:
            raise ValueError("Unsupported `width`.")
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        self.conv_2d = nn.Conv2d(
            num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        if depth == 18:
            num_blocks = (2, 2, 2, 2)
        elif depth == 34:
            num_blocks = (3, 4, 6, 3)
        else:
            raise ValueError("Unsupported `depth`.")
        self.layer_0 = self._make_layer(64, 64, num_blocks[0], 1, activation_fn)
        self.layer_1 = self._make_layer(64, 128, num_blocks[1], 2, activation_fn)
        self.layer_2 = self._make_layer(128, 256, num_blocks[2], 2, activation_fn)
        self.layer_3 = self._make_layer(256, 512, num_blocks[3], 2, activation_fn)
        self.batchnorm = nn.BatchNorm2d(512)
        self.relu = activation_fn()
        self.logits = nn.Linear(512, num_classes)
        if adaptive_pooling:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AvgPool2d(4)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, activation_fn):
        layers = []
        for i, stride in enumerate([stride] + [1] * (num_blocks - 1)):
            layers.append(
                _PreActBlock(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    stride,
                    activation_fn,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std
        out = self.conv_2d(out)
        out = self.layer_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.relu(self.batchnorm(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.logits(out)

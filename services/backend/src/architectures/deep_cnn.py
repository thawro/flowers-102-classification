from torch.nn.common_types import _size_2_t
from collections import OrderedDict
from typing import Literal, Optional
import torch
from torch import nn


class CNNBlock(nn.Module):
    """Single CNN block constructed of combination of Conv2d, Activation, Pooling, Batch Normalization and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        groups: int = 1,
        activation: Optional[str] = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of Conv2d input channels.
            out_channels (int): Number of Conv2d out channels.
            kernel_size (int): Conv2d kernel equal to `(kernel_size, kernel_size)`.
            stride (int, optional): Conv2d stride equal to `(stride, stride)`.
                Defaults to 1.
            padding (int | str, optional): Conv2d padding equal to `(padding, padding)`.
                Defaults to 1.. Defaults to 0.
            activation (str, optional): Type of activation function used before BN. Defaults to 0.
        """
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation_fn = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation_fn(out)
        return out


class DeepCNN(nn.Module):
    """Deep Convolutional Neural Network (CNN) constructed of many CNN blocks and ended with Global Average Pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernels: list[_size_2_t],
        pool_type: Literal["Max", "Avg"] = "Max",
        activation: str = "ReLU",
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            out_channels (list[int]): Number of channels used in CNN blocks.
            kernels (int | list[int]): Kernels of Conv2d in CNN blocks.
                If int or tuple[int, int] is passed, then all layers use same kernel size.
            pool_kernels (int | list[int]): Kernels of Pooling in CNN blocks.
                If int is passed, then all layers use same pool kernel size.
            pool_type (Literal["Max", "Avg"], optional): Pooling type in CNN blocks. Defaults to "Max".
            activation (str, optional): Type of activation function used in CNN blocks. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernels = kernels
        self.pool_type = pool_type
        self.activation = activation
        n_blocks = len(out_channels)
        if isinstance(kernels, int) or isinstance(kernels, tuple):
            kernels = [kernels] * n_blocks
        layers: list[tuple[str, nn.Module]] = [
            (
                f"conv_{i}",
                CNNBlock(
                    in_channels if i == 0 else out_channels[i - 1],
                    out_channels[i],
                    kernels[i],
                    activation=activation,
                ),
            )
            for i in range(n_blocks)
        ]
        self.net = nn.Sequential(OrderedDict(layers))
        self.out_channels = self.out_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def name(self):
        return "DeepCNN"

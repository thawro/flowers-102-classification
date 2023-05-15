"""Implementation based on https://arxiv.org/pdf/1512.03385.pdf
with modifications suggested in https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""
from functools import partial
from typing import Literal
from collections import OrderedDict
import torchvision
from torch import nn
import torch
from src.architectures.deep_cnn import CNNBlock
from src.architectures.base import Backbone
from torch.nn.common_types import _size_2_t


class BasicBlock(nn.Module):
    """Basic Residual Block.
    Pipeline without downsampling:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU(out + x)

    If downsampling is present, then shortcut connection is added:
    Shortcut = Conv1x1 -> BN
    x -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU(out + Shortcut(x))
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        downsample: bool = False,
    ):
        """
        Args:
            in_channels (int): Number of block input channels.
            out_channels (int): Number of block output channels.
            kernel_size (_size_2_t): Kernel used for both Conv2d layers.
            stride (_size_2_t): Stride used in first Conv2d layer and in optional shortcut.
            downsample (bool, optional): Whether to apply downsampling. Defaults to False.
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.shortcut = CNNBlock(
                in_channels, out_channels, kernel_size=1, stride=stride, activation="Identity"
            )
        else:
            out_channels = in_channels

        self.conv1 = CNNBlock(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1
        )

        self.conv2 = CNNBlock(
            out_channels, out_channels, kernel_size=kernel_size, padding=1, activation="Identity"
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    """Bottleneck Residual Block.
    Pipeline without downsampling:
    x -> Conv1x1 -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv1x1 -> BN -> ReLU(out + x)

    If downsampling is present, then shortcut connection is added:
    Shortcut = Conv1x1 -> BN
    x -> Conv1x1 -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv1x1 -> BN -> ReLU(out + Shortcut(x))
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        downsample: bool = False,
        groups: int = 1,
    ):
        """
        Args:
            in_channels (int): Number of block input channels.
            mid_channels (int): Number of block bottleneck channels.
            kernel_size (_size_2_t): Kernel used for middle Conv2d layer.
            stride (_size_2_t): Stride used in middle Conv2d layer and in optional shortcut.
            downsample (bool, optional): Whether to apply downsampling. Defaults to False.
        """

        super().__init__()
        self.downsample = downsample
        if self.downsample:
            out_channels = mid_channels * self.expansion
            self.shortcut = CNNBlock(
                in_channels, out_channels, kernel_size=1, stride=stride, activation="Identity"
            )
        else:
            out_channels = in_channels
        self.conv1 = CNNBlock(in_channels, mid_channels, kernel_size=1)
        self.conv2 = CNNBlock(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
            groups=groups,
        )
        self.conv3 = CNNBlock(mid_channels, out_channels, kernel_size=1, activation="Identity")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            identity = self.shortcut(x)
        else:
            identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.relu(out + identity)


class BasicResNetCore(nn.Module):
    """Core Basic Residual layers"""

    def __init__(
        self,
        in_channels: int,
        stages_n_blocks: list[int],
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
        """
        super().__init__()
        out_channels = in_channels
        layers: list[tuple[str, nn.Module]] = []
        for stage, n_blocks in enumerate(stages_n_blocks):
            is_first_stage = stage == 0
            blocks_layers = []
            for i in range(n_blocks):
                is_first_block = i == 0
                downsample = not is_first_stage and is_first_block
                stride = 2 if downsample else 1
                block = BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    kernel_size=3,
                    downsample=downsample,
                )
                if not is_first_stage:
                    in_channels = out_channels
                blocks_layers.append(block)
            in_channels, out_channels = out_channels, in_channels * 2
            blocks_layers = nn.Sequential(*blocks_layers)
            layers.append((f"stage_{stage}", blocks_layers))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottleneckResNetCore(nn.Module):
    """Core Bottleneck Residual layers"""

    def __init__(
        self,
        in_channels: int,
        first_conv_out_channels: int,
        stages_n_blocks: list[int],
        groups: int = 1,
        block_class=BottleneckBlock,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            first_conv_out_channels (int): TODO
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
        """
        super().__init__()
        layers: list[tuple[str, nn.Module]] = []
        mid_channels = first_conv_out_channels
        for stage, n_blocks in enumerate(stages_n_blocks):
            is_first_stage = stage == 0
            blocks_layers = []
            for i in range(n_blocks):
                is_first_block = i == 0
                stride = 2 if not is_first_stage and is_first_block else 1
                block = block_class(
                    in_channels=in_channels,
                    mid_channels=mid_channels,
                    kernel_size=3,
                    stride=stride,
                    downsample=is_first_block,
                    groups=groups,
                )
                in_channels = mid_channels * block.expansion
                blocks_layers.append(block)
            mid_channels *= 2
            blocks_layers = nn.Sequential(*blocks_layers)
            layers.append((f"stage_{stage}", blocks_layers))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseResNet(Backbone):
    """Convolutional Neural Network with residual connextions"""

    def __init__(
        self,
        in_channels: int,
        block_type: Literal["basic", "bottleneck"],
        stages_n_blocks: list[int],
        stem_channels: int = 64,
        stem_kernel_size: _size_2_t = 7,
        pool_kernel_size: _size_2_t = 3,
    ):
        """
        Args:
            in_channels (int): Number of image channels.
            stem_channels (int): Number of channels used in first Conv2d layer (before the residual layers).
            stem_kernel_size (int): Kernel used in first Conv2d layer
                (before the residual layers) as '(stem_kernel_size, stem_kernel_size)`.
            pool_kernel_size (int): Kernel used in Pooling layer after the first Conv2d layer
                (before the residual layers) as '(pool_kernel_size, pool_kernel_size)`.
            stages_n_blocks (list[int]): Number of bottleneck blocks used per stage.
            block_type (Literal["basic", "bottleneck"], optional): Whether to use Basic block or Bottleneck block.
                Defaults to "basic".
        """
        self.stem_channels = stem_channels
        self.stem_kernel_size = stem_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.block_type = block_type
        self.stages_n_blocks = stages_n_blocks
        ResnetCoreBlocks = (
            BasicResNetCore
            if block_type == "basic"
            else partial(BottleneckResNetCore, first_conv_out_channels=stem_channels)
        )
        layers = [
            CNNBlock(
                in_channels,
                stem_channels,
                stem_kernel_size,
                stride=2,
                padding=3,
            ),
            nn.MaxPool2d(pool_kernel_size, stride=2),
            ResnetCoreBlocks(in_channels=stem_channels, stages_n_blocks=stages_n_blocks),
        ]
        # layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(1, -1)])
        net = nn.Sequential(*layers)
        super().__init__(net, self.out_channels, self.__class__.__name__)

    @property
    def out_channels(self) -> int:
        n_stages = len(self.stages_n_blocks)
        if self.block_type == "basic":
            return self.stem_channels * 2 ** (n_stages - 1)
        else:
            return self.stem_channels * 2 ** (n_stages - 1) * BottleneckBlock.expansion

    @property
    def name(self):
        return self.__class__.__name__


class ResNet18(BaseResNet):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stages_n_blocks=[2, 2, 2, 2],
            block_type="basic",
        )


class ResNet34(BaseResNet):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stages_n_blocks=[3, 4, 6, 3],
            block_type="basic",
        )


class ResNet50(BaseResNet):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stages_n_blocks=[3, 4, 6, 3],
            block_type="bottleneck",
        )


class ResNet101(BaseResNet):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stages_n_blocks=[3, 4, 23, 3],
            block_type="bottleneck",
        )


class ResNet152(BaseResNet):
    def __init__(self, in_channels: int):
        super().__init__(
            in_channels=in_channels,
            stages_n_blocks=[3, 8, 36, 3],
            block_type="bottleneck",
        )


class ResNet:
    def __new__(
        cls,
        in_channels: int,
        version: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        version2cfg = {
            # local_class, torch_load_fn, out_channels
            "resnet18": (ResNet18, torchvision.models.resnet18, 512),
            "resnet34": (ResNet34, torchvision.models.resnet34, 512),
            "resnet50": (ResNet50, torchvision.models.resnet50, 2048),
            "resnet101": (ResNet101, torchvision.models.resnet101, 2048),
            "resnet152": (ResNet152, torchvision.models.resnet152, 2048),
        }
        ModelClass, torch_load_fn, out_channels = version2cfg[version]
        if load_from_torch:
            params = dict(pretrained=pretrained) if pretrained else dict()
            _net = torch_load_fn(**params)
            _net.fc = nn.Identity()
            net = Backbone(_net, out_channels, version)
            if freeze_extractor:
                net.freeze()
        else:
            net = ModelClass(in_channels=in_channels)
        return net

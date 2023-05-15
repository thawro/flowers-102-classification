"""SqueezeNet architecture based on https://arxiv.org/pdf/1602.07360.pdf.
By default, the simple bypass version is used
Also, BatchNormalization is added for each squeeze and expand layers"""

import torch
from torch import nn
from collections import OrderedDict
from src.architectures.deep_cnn import CNNBlock
from src.architectures.base import Backbone


class FireBlock(nn.Module):
    """FireBlock used to squeeze and expand convolutional channels"""

    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float,
        expand_filters: int,
        pct_3x3: float,
        is_residual: bool = False,
    ):
        super().__init__()
        s_1x1 = int(squeeze_ratio * expand_filters)
        e_3x3 = int(expand_filters * pct_3x3)
        e_1x1 = expand_filters - e_3x3
        self.squeeze_1x1 = CNNBlock(in_channels, s_1x1, kernel_size=1)
        self.expand_1x1 = CNNBlock(s_1x1, e_1x1, kernel_size=1)
        self.expand_3x3 = CNNBlock(s_1x1, e_3x3, kernel_size=3, padding=1)
        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_out = self.squeeze_1x1(x)
        expand_1x1_out = self.expand_1x1(squeeze_out)
        expand_3x3_out = self.expand_3x3(squeeze_out)
        out = torch.concat([expand_1x1_out, expand_3x3_out], dim=1)  # concat over channels
        if self.is_residual:
            return x + out
        return out


class SqueezeNet1_0(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_e: int = 128,
        incr_e: int = 128,
        pct_3x3: float = 0.5,
        freq: int = 2,
        SR: float = 0.125,
        simple_bypass: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_e = base_e
        self.incr_e = incr_e
        self.pct_3x3 = pct_3x3
        self.freq = freq
        self.SR = SR

        # architecture, fb - fire block
        out_channels = 96
        n_fire_blocks = 8
        fb_expand_filters = [base_e + (incr_e * (i // freq)) for i in range(n_fire_blocks)]
        fb_in_channels = [out_channels] + fb_expand_filters
        is_residual = [False] + [(i % freq == 1 and simple_bypass) for i in range(1, n_fire_blocks)]
        self.fb_in_channels = fb_in_channels
        self.out_channels = fb_expand_filters[-1]
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2)
        maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire2 = FireBlock(fb_in_channels[0], SR, fb_expand_filters[0], pct_3x3, is_residual[0])
        fire3 = FireBlock(fb_in_channels[1], SR, fb_expand_filters[1], pct_3x3, is_residual[1])
        fire4 = FireBlock(fb_in_channels[2], SR, fb_expand_filters[2], pct_3x3, is_residual[2])
        maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire5 = FireBlock(fb_in_channels[3], SR, fb_expand_filters[3], pct_3x3, is_residual[3])
        fire6 = FireBlock(fb_in_channels[4], SR, fb_expand_filters[4], pct_3x3, is_residual[4])
        fire7 = FireBlock(fb_in_channels[5], SR, fb_expand_filters[5], pct_3x3, is_residual[5])
        fire8 = FireBlock(fb_in_channels[6], SR, fb_expand_filters[6], pct_3x3, is_residual[6])
        maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        fire9 = FireBlock(fb_in_channels[7], SR, fb_expand_filters[7], pct_3x3, is_residual[7])
        dropout9 = nn.Dropout2d(p=0.5)
        layers = [
            ("conv1", conv1),
            ("maxpool1", maxpool1),
            ("fire2", fire2),
            ("fire3", fire3),
            ("fire4", fire4),
            ("maxpool4", maxpool4),
            ("fire5", fire5),
            ("fire6", fire6),
            ("fire7", fire7),
            ("fire8", fire8),
            ("maxpool8", maxpool8),
            ("fire9", fire9),
            ("dropout9", dropout9),
        ]
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def name(self):
        return "SqueezeNet"


from typing import Literal


class SqueezeNet:
    def __new__(
        cls,
        in_channels: int,
        version: Literal["squeezenet1_0", "squeezenet1_1"],
        load_from_torch: bool = False,
        pretrained: bool = False,
        freeze_extractor: bool = False,
    ):
        if load_from_torch:
            _net = torch.hub.load("pytorch/vision:v0.10.0", version, pretrained=pretrained)
            _net.classifier[0] = torch.nn.Identity()
            _net.classifier[1] = torch.nn.Identity()
            _net.classifier[2] = torch.nn.Identity()
            net = Backbone(_net, out_channels=512, name=version)
            if freeze_extractor:
                net.freeze()
        else:
            net = SqueezeNet1_0(in_channels=in_channels)
        return net

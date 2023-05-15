from torch import nn
import torch


class Backbone(nn.Module):
    def __init__(self, net: nn.Module, out_channels: int, name: str):
        super().__init__()
        self.net = net
        self.out_channels = out_channels
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

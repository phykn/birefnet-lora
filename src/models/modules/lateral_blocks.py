import torch
import torch.nn as nn


class BasicLatBlk(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        ks: int = 1,
        s: int = 1,
        p: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = ks,
            stride = s,
            padding = p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
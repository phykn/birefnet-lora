import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank

        for param in self.linear.parameters():
            param.requires_grad = False

        self.down = nn.Linear(
            in_features=linear.in_features,
            out_features=rank,
            bias=False,
        )
        self.up = nn.Linear(
            in_features=rank,
            out_features=linear.out_features,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.up(self.down(x)) * self.scale


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.conv = conv
        self.scale = alpha / rank

        for param in self.conv.parameters():
            param.requires_grad = False

        if conv.groups != 1:
            raise ValueError("LoRAConv2d currently supports grouped conv only when groups=1.")

        self.down = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.up = nn.Conv2d(
            in_channels=rank,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.up(self.down(x)) * self.scale


def apply_linear(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> None:
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear):
                setattr(
                    module,
                    child_name,
                    LoRALinear(linear=child_module, rank=rank, alpha=alpha),
                )


def apply_conv2d(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    exclude_names: list[str] | None = None,
) -> None:
    excludes = exclude_names or []
    for _, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Conv2d):
                if any(name in child_name for name in excludes):
                    continue
                setattr(
                    module,
                    child_name,
                    LoRAConv2d(conv=child_module, rank=rank, alpha=alpha),
                )

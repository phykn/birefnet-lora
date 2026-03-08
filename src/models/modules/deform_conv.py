import torch
import torch.nn as nn

from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        bias: bool = False
    ) -> None:
        super().__init__()

        assert isinstance(kernel_size, (tuple, int))

        k_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 2 * k_size[0] * k_size[1],
            kernel_size = k_size,
            stride = stride,
            padding = self.padding,
            bias = True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 1 * k_size[0] * k_size[1],
            kernel_size = k_size,
            stride = stride,
            padding = self.padding,
            bias = True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = k_size,
            stride = stride,
            padding = self.padding,
            bias = bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        return deform_conv2d(
            input = x,
            offset = offset,
            weight = self.regular_conv.weight,
            bias = self.regular_conv.bias,
            padding = self.padding,
            mask = modulator,
            stride = self.stride,
        )

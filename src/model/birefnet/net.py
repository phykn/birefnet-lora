import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import laplacian

from .decoder.block import BasicDecBlk
from .decoder.net import Decoder
from .swin import build_large


class BiRefNet(nn.Module):
    def __init__(
        self,
        channels: list[int] = [1536, 768, 384, 192],
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        channels = [channel * 2 for channel in channels]

        self.cxt = channels[1:][::-1][-3:]

        self.bb = build_large(grad_checkpoint=grad_checkpoint)

        self.squeeze_module = nn.Sequential(
            BasicDecBlk(channels[0] + sum(self.cxt), channels[0])
        )

        self.decoder = Decoder(
            channels=channels,
        )

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4 = self.bb(x)

        B, C, H, W = x.shape
        x_pyramid = F.interpolate(
            x, size=(H // 2, W // 2), mode="bilinear", align_corners=True
        )
        pyramid_x1, pyramid_x2, pyramid_x3, pyramid_x4 = self.bb(x_pyramid)
        x1 = torch.cat(
            [
                x1,
                F.interpolate(
                    pyramid_x1,
                    size=x1.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x2 = torch.cat(
            [
                x2,
                F.interpolate(
                    pyramid_x2,
                    size=x2.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x3 = torch.cat(
            [
                x3,
                F.interpolate(
                    pyramid_x3,
                    size=x3.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )
        x4 = torch.cat(
            [
                x4,
                F.interpolate(
                    pyramid_x4,
                    size=x4.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                ),
            ],
            dim=1,
        )

        if self.cxt:
            cxt_size = len(self.cxt)
            interpolated_features = [
                F.interpolate(
                    x1, size=x4.shape[2:], mode="bilinear", align_corners=True
                ),
                F.interpolate(
                    x2, size=x4.shape[2:], mode="bilinear", align_corners=True
                ),
                F.interpolate(
                    x3, size=x4.shape[2:], mode="bilinear", align_corners=True
                ),
            ]

            x4 = torch.cat((*interpolated_features[-cxt_size:], x4), dim=1)

        return (x1, x2, x3, x4)

    def predict(
        self, x: torch.Tensor
    ) -> tuple[
        list[torch.Tensor] | tuple[list[list[torch.Tensor]], list[torch.Tensor]],
        None,
    ]:
        x1, x2, x3, x4 = self.encode(x)

        x4 = self.squeeze_module(x4)

        features = [x, x1, x2, x3, x4]

        if self.training:
            features.append(laplacian(torch.mean(x, dim=1).unsqueeze(1), kernel_size=5))

        scaled_preds = self.decoder(features)

        return scaled_preds, None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor] | list[object]:
        scaled_preds, class_preds = self.predict(x)
        class_preds_lst = [class_preds]
        return [scaled_preds, class_preds_lst] if self.training else scaled_preds

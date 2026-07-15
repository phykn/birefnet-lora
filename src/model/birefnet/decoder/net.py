import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .block import BasicDecBlk, BasicLatBlk

GDT_INTER_CHANNELS = 16


def split_patches(
    image: torch.Tensor,
    grid_h: int = 2,
    grid_w: int = 2,
    patch_ref: torch.Tensor | None = None,
    transformation: str = "b c (hg h) (wg w) -> (b hg wg) c h w",
) -> torch.Tensor:
    if patch_ref is not None:
        grid_h = image.shape[-2] // patch_ref.shape[-2]
        grid_w = image.shape[-1] // patch_ref.shape[-1]

    return rearrange(image, pattern=transformation, hg=grid_h, wg=grid_w)


class SimpleConvs(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels: int = 64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv1(x))


class Decoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
    ) -> None:
        super().__init__()

        ipt_blk_in_channels = [3072, 768, 192, 48, 3]
        ipt_blk_out_channels = [channels[i] // 8 for i in range(4)]

        self.ipt_blk5 = SimpleConvs(
            ipt_blk_in_channels[0], ipt_blk_out_channels[0], inter_channels=64
        )
        self.ipt_blk4 = SimpleConvs(
            ipt_blk_in_channels[1], ipt_blk_out_channels[0], inter_channels=64
        )
        self.ipt_blk3 = SimpleConvs(
            ipt_blk_in_channels[2], ipt_blk_out_channels[1], inter_channels=64
        )
        self.ipt_blk2 = SimpleConvs(
            ipt_blk_in_channels[3], ipt_blk_out_channels[2], inter_channels=64
        )
        self.ipt_blk1 = SimpleConvs(
            ipt_blk_in_channels[4], ipt_blk_out_channels[3], inter_channels=64
        )

        bb_neck_out_channels = channels.copy()

        dec_blk_out_channels = [c for c in bb_neck_out_channels[1:]] + [
            bb_neck_out_channels[-1] // 2
        ]

        dec_blk_in_channels = [
            bb_neck_out_channels[i] + ipt_blk_out_channels[max(0, i - 1)]
            for i in range(len(bb_neck_out_channels))
        ]

        self.decoder_block4 = BasicDecBlk(
            dec_blk_in_channels[0], dec_blk_out_channels[0]
        )
        self.decoder_block3 = BasicDecBlk(
            dec_blk_in_channels[1], dec_blk_out_channels[1]
        )
        self.decoder_block2 = BasicDecBlk(
            dec_blk_in_channels[2], dec_blk_out_channels[2]
        )
        self.decoder_block1 = BasicDecBlk(
            dec_blk_in_channels[3], dec_blk_out_channels[3]
        )

        conv_out1_in_channels = dec_blk_out_channels[3] + ipt_blk_out_channels[3]
        self.conv_out1 = nn.Sequential(nn.Conv2d(conv_out1_in_channels, 1, 1))

        self.lateral_block4 = BasicLatBlk(
            bb_neck_out_channels[1], dec_blk_out_channels[0]
        )
        self.lateral_block3 = BasicLatBlk(
            bb_neck_out_channels[2], dec_blk_out_channels[1]
        )
        self.lateral_block2 = BasicLatBlk(
            bb_neck_out_channels[3], dec_blk_out_channels[2]
        )

        self.conv_ms_spvn_4 = nn.Conv2d(dec_blk_out_channels[0], 1, 1)
        self.conv_ms_spvn_3 = nn.Conv2d(dec_blk_out_channels[1], 1, 1)
        self.conv_ms_spvn_2 = nn.Conv2d(dec_blk_out_channels[2], 1, 1)

        self.gdt_convs_4 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[0], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_3 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[1], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.gdt_convs_2 = nn.Sequential(
            nn.Conv2d(dec_blk_out_channels[2], GDT_INTER_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(GDT_INTER_CHANNELS),
            nn.ReLU(inplace=True),
        )

        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))

        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(GDT_INTER_CHANNELS, 1, 1))

    def forward(
        self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor] | tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        if self.training:
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features

        outs = []

        patches_batch = split_patches(
            x,
            patch_ref=x4,
            transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
        )
        _patched = self.ipt_blk5(
            F.interpolate(
                patches_batch,
                size=x4.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        x4 = torch.cat((x4, _patched), dim=1)

        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.training else None

        p4_gdt = self.gdt_convs_4(p4)
        if self.training:
            m4_dia = m4
            gdt_label_main_4 = gdt_gt * F.interpolate(
                m4_dia,
                size=gdt_gt.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            outs_gdt_label.append(gdt_label_main_4)
            gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
            outs_gdt_pred.append(gdt_pred_4)
        gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
        p4 = p4 * gdt_attn_4

        _p4 = F.interpolate(
            p4,
            size=x3.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p3 = _p4 + self.lateral_block4(x3)

        patches_batch = split_patches(
            x,
            patch_ref=_p3,
            transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
        )
        _patched = self.ipt_blk4(
            F.interpolate(
                patches_batch,
                size=x3.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        _p3 = torch.cat((_p3, _patched), dim=1)

        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.training else None

        p3_gdt = self.gdt_convs_3(p3)
        if self.training:
            m3_dia = m3
            gdt_label_main_3 = gdt_gt * F.interpolate(
                m3_dia,
                size=gdt_gt.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            outs_gdt_label.append(gdt_label_main_3)
            gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
            outs_gdt_pred.append(gdt_pred_3)
        gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
        p3 = p3 * gdt_attn_3

        _p3 = F.interpolate(
            p3,
            size=x2.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p2 = _p3 + self.lateral_block3(x2)

        patches_batch = split_patches(
            x,
            patch_ref=_p2,
            transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
        )
        _patched = self.ipt_blk3(
            F.interpolate(
                patches_batch,
                size=x2.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        _p2 = torch.cat((_p2, _patched), dim=1)

        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.training else None

        p2_gdt = self.gdt_convs_2(p2)
        if self.training:
            m2_dia = m2
            gdt_label_main_2 = gdt_gt * F.interpolate(
                m2_dia,
                size=gdt_gt.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            outs_gdt_label.append(gdt_label_main_2)
            gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
            outs_gdt_pred.append(gdt_pred_2)
        gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
        p2 = p2 * gdt_attn_2

        _p2 = F.interpolate(
            p2,
            size=x1.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        _p1 = _p2 + self.lateral_block2(x1)

        patches_batch = split_patches(
            x,
            patch_ref=_p1,
            transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
        )
        _patched = self.ipt_blk2(
            F.interpolate(
                patches_batch,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        _p1 = torch.cat((_p1, _patched), dim=1)

        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(
            _p1,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        patches_batch = split_patches(
            x,
            patch_ref=_p1,
            transformation="b c (hg h) (wg w) -> b (c hg wg) h w",
        )
        _patched = self.ipt_blk1(
            F.interpolate(
                patches_batch,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        )
        _p1 = torch.cat((_p1, _patched), dim=1)

        p1_out = self.conv_out1(_p1)

        if self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
        outs.append(p1_out)

        if not self.training:
            return outs

        return [outs_gdt_pred, outs_gdt_label], outs

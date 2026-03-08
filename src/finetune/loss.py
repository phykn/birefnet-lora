import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        dims = (1, 2, 3)
        inter = torch.sum(target * pred, dim=dims)
        union = torch.sum(target, dim=dims) + torch.sum(pred, dim=dims) - inter
        iou = 1 - inter / (union + 1e-6)
        return iou.mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 30.0,
        lambda_iou: float = 0.5
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()
        self.lambda_bce = lambda_bce
        self.lambda_iou = lambda_iou

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        target = torch.clamp(target, 0, 1)

        loss_bce = self.bce(pred, target) * self.lambda_bce
        loss_iou = self.iou(pred.sigmoid(), target) * self.lambda_iou

        return loss_bce + loss_iou

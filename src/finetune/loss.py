import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduce_dims = (1, 2, 3)
        intersection = torch.sum(target * pred, dim=reduce_dims)
        union = torch.sum(target, dim=reduce_dims) + torch.sum(pred, dim=reduce_dims) - intersection
        iou_loss = 1 - intersection / (union + 1e-6)
        return iou_loss.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, lambda_bce: float = 30.0, lambda_iou: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()
        self.lambda_bce = lambda_bce
        self.lambda_iou = lambda_iou

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        target = torch.clamp(target, 0, 1)

        bce_loss = self.bce(pred, target) * self.lambda_bce
        iou_loss = self.iou(pred.sigmoid(), target) * self.lambda_iou

        return bce_loss + iou_loss


class ConsistencyLoss(nn.Module):
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        if logits1.shape[2:] != logits2.shape[2:]:
            logits2 = F.interpolate(
                logits2,
                size=logits1.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
        return F.mse_loss(logits1, logits2)

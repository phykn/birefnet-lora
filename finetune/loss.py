"""
세그먼테이션 Loss.

원본 PixLoss의 핵심 구성(BCE + IoU)을 유지하되, 단일 출력에 대해서만 계산한다.
원본과 동일하게 pred.sigmoid() 후 BCELoss를 사용한다 (BCEWithLogitsLoss 아님에 주의).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """원본 loss.py의 IoULoss와 동일."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = pred.shape[0]
        loss = 0.0
        for i in range(b):
            intersection = torch.sum(target[i] * pred[i])
            union = torch.sum(target[i]) + torch.sum(pred[i]) - intersection
            loss += 1 - intersection / union
        return loss


class SegmentationLoss(nn.Module):
    """
    예측 마스크 vs 정답 마스크 Loss.

    Args:
        lambda_bce: BCE Loss 가중치 (원본 기본값 30)
        lambda_iou: IoU Loss 가중치 (원본 기본값 0.5)
    """

    def __init__(self, lambda_bce=30.0, lambda_iou=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.iou = IoULoss()
        self.lambda_bce = lambda_bce
        self.lambda_iou = lambda_iou

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 1, H, W] 모델 출력 logit (sigmoid 전)
            target: [B, 1, H, W] 정답 마스크, 값 범위 0~1
        """
        # 해상도 맞추기
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=True)

        # ★ 원본과 동일: sigmoid 먼저 적용 후 BCELoss
        pred_sigmoid = pred.sigmoid()
        target = torch.clamp(target, 0, 1)

        loss_bce = self.bce(pred_sigmoid, target) * self.lambda_bce
        loss_iou = self.iou(pred_sigmoid, target) * self.lambda_iou

        return loss_bce + loss_iou

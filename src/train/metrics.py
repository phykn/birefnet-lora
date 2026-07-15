import cv2
import numpy as np
import torch


def score_brier(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    error = (torch.sigmoid(logits) - target.clamp(0, 1)).square() * valid_mask
    return error.sum() / valid_mask.sum().clamp_min(1.0)


def score_ece(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    bins: int = 10,
) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    active = valid_mask > 0.5
    probability = torch.sigmoid(logits)[active]
    labels = target[active].clamp(0, 1)
    if probability.numel() == 0:
        return logits.sum() * 0.0

    ece = probability.new_zeros(())
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    for index in range(bins):
        upper = (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        in_bin = (probability >= edges[index]) & upper
        count = in_bin.sum()
        if count > 0:
            confidence = probability[in_bin].mean()
            accuracy = labels[in_bin].mean()
            ece = ece + count / probability.numel() * (confidence - accuracy).abs()
    return ece


def score_iou(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    pred = (probability >= threshold).to(target.dtype) * valid_mask
    target = (target > 0.5).to(target.dtype) * valid_mask
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    scores = torch.where(
        union == 0,
        torch.ones_like(union),
        intersection / union.clamp_min(1.0),
    )
    return scores.mean()


def score_iou_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    return score_iou(
        torch.sigmoid(logits), target, valid_mask, threshold
    )


def score_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    pred = (torch.sigmoid(logits) >= threshold).to(target.dtype) * valid_mask
    target = (target > 0.5).to(target.dtype) * valid_mask
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    total = pred.sum(dim=dims) + target.sum(dim=dims)
    scores = torch.where(
        total == 0,
        torch.ones_like(total),
        2.0 * intersection / total.clamp_min(1.0),
    )
    return scores.mean()


def _find_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        binary,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return binary - eroded


def score_boundary(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: float = 2.0,
) -> float:
    pred_boundary = _find_boundary(prediction)
    target_boundary = _find_boundary(target)
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0

    distance_to_target = cv2.distanceTransform(
        1 - target_boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_to_pred = cv2.distanceTransform(
        1 - pred_boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    precision = (
        float(
            np.logical_and(pred_boundary > 0, distance_to_target <= tolerance_px).sum()
        )
        / pred_count
    )
    recall = (
        float(
            np.logical_and(target_boundary > 0, distance_to_pred <= tolerance_px).sum()
        )
        / target_count
    )
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_boundary_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    tolerance_px: float = 2.0,
) -> float:
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    pred = ((torch.sigmoid(logits) >= threshold) * (valid_mask > 0.5)).cpu().numpy()
    gt = ((target > 0.5) * (valid_mask > 0.5)).cpu().numpy()
    return float(
        np.mean(
            [
                score_boundary(p[0], t[0], tolerance_px=tolerance_px)
                for p, t in zip(pred, gt)
            ]
        )
    )

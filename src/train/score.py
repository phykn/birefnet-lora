import cv2
import numpy as np
import torch


def brier(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(target)
    error = (torch.sigmoid(logits) - target.clamp(0, 1)).square() * valid
    return error.sum() / valid.sum().clamp_min(1.0)


def ece(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    bins: int = 10,
) -> torch.Tensor:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if valid is None:
        valid = torch.ones_like(target)
    active = valid > 0.5
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


def iou(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(target)
    pred = (probability >= threshold).to(target.dtype) * valid
    target = (target > 0.5).to(target.dtype) * valid
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    scores = torch.where(
        union == 0,
        torch.ones_like(union),
        intersection / union.clamp_min(1.0),
    )
    return scores.mean()


def iou_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    return iou(torch.sigmoid(logits), target, valid, threshold)


def dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(target)
    pred = (torch.sigmoid(logits) >= threshold).to(target.dtype) * valid
    target = (target > 0.5).to(target.dtype) * valid
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    total = pred.sum(dim=dims) + target.sum(dim=dims)
    scores = torch.where(
        total == 0,
        torch.ones_like(total),
        2.0 * intersection / total.clamp_min(1.0),
    )
    return scores.mean()


def _find_edge(mask: np.ndarray) -> np.ndarray:
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


def boundary(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: float = 2.0,
) -> float:
    pred_boundary = _find_edge(prediction)
    target_boundary = _find_edge(target)
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


def boundary_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold: float = 0.5,
    tolerance_px: float = 2.0,
) -> float:
    if valid is None:
        valid = torch.ones_like(target)
    pred = ((torch.sigmoid(logits) >= threshold) * (valid > 0.5)).cpu().numpy()
    gt = ((target > 0.5) * (valid > 0.5)).cpu().numpy()
    return float(
        np.mean(
            [
                boundary(p[0], t[0], tolerance_px=tolerance_px)
                for p, t in zip(pred, gt)
            ]
        )
    )

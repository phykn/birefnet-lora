import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.clamp(0, 1)
    if valid is None:
        valid = torch.ones_like(target)
    else:
        valid = valid.clamp(0, 1)
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode="nearest")
        valid = F.interpolate(valid, size=pred.shape[2:], mode="nearest")
    return target, valid


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denom = valid.sum().clamp_min(1.0)
    return (value * valid).sum() / denom


def erode_valid(valid: torch.Tensor, radius: int) -> torch.Tensor:
    invalid = 1.0 - valid
    kernel = radius * 2 + 1
    return 1.0 - F.max_pool2d(
        invalid,
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )


class GCELoss(nn.Module):
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError("GCE q must be in (0, 1]")
        self.q = float(q)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        if weight is None:
            weight = torch.ones_like(target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = -torch.expm1(-self.q * bce) / self.q
        return masked_mean(loss * weight, valid)


class IoULoss(nn.Module):
    """Sample-wise soft IoU with an explicit empty-empty=perfect contract."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target * valid).sum(dim=dims)
        union = ((pred + target - pred * target) * valid).sum(dim=dims)
        loss = torch.where(
            union <= self.eps,
            torch.zeros_like(union),
            1.0 - (intersection + self.eps) / (union + self.eps),
        )
        return loss.mean()


class DiceLoss(nn.Module):
    """Sample-wise soft Dice with an explicit empty-empty=perfect contract."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target * valid).sum(dim=dims)
        total = ((pred + target) * valid).sum(dim=dims)
        loss = torch.where(
            total <= self.eps,
            torch.zeros_like(total),
            1.0 - (2.0 * intersection + self.eps) / (total + self.eps),
        )
        return loss.mean()


def make_band(target: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        raise ValueError("boundary radius must be >= 1")
    kernel = radius * 2 + 1
    padded = F.pad(target, (radius, radius, radius, radius), value=0.0)
    dilated = F.max_pool2d(padded, kernel_size=kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel_size=kernel, stride=1)
    return (dilated - eroded > 0).to(target.dtype)


class BoundaryBCELoss(nn.Module):
    def __init__(self, radius: int = 3) -> None:
        super().__init__()
        self.radius = int(radius)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor | None = None,
        cut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = make_band(target, self.radius) * valid
        if cut is not None:
            if cut.shape[2:] != target.shape[2:]:
                cut = F.interpolate(cut, size=target.shape[2:], mode="nearest")
            # Only augmentation cuts are excluded; original image edges stay valid.
            blocked = F.max_pool2d(
                cut.clamp(0, 1),
                kernel_size=self.radius * 2 + 1,
                stride=1,
                padding=self.radius,
            )
            active = active * (1.0 - blocked)
        if active.sum() <= 0:
            return pred.sum() * 0.0
        if weight is None:
            weight = torch.ones_like(target)
        pixel_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return masked_mean(pixel_bce * weight, active)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        gce_q: float = 0.7,
        lambda_cls: float = 1.0,
        lambda_region: float = 1.0,
        lambda_boundary: float = 1.0,
        region_loss: str = "dice",
        boundary_radius: int = 3,
    ) -> None:
        super().__init__()
        if region_loss not in {"dice", "iou"}:
            raise ValueError("region_loss must be 'dice' or 'iou'")
        self.cls = GCELoss(q=gce_q)
        self.region = DiceLoss() if region_loss == "dice" else IoULoss()
        self.boundary = BoundaryBCELoss(radius=boundary_radius)
        self.lambda_cls = float(lambda_cls)
        self.lambda_region = float(lambda_region)
        self.lambda_boundary = float(lambda_boundary)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        cut: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> dict[str, torch.Tensor]:
        target, valid = _resize(pred, target, valid)
        if weight is None:
            weight = torch.ones_like(target)
        elif weight.shape[2:] != pred.shape[2:]:
            weight = F.interpolate(weight, size=pred.shape[2:], mode="area")
        weight = weight.clamp(0, 1)
        raw_cls = self.cls(pred, target, valid, weight)
        raw_region = self.region(pred.sigmoid(), target, valid * weight)
        raw_boundary = (
            self.boundary(pred, target, valid, weight, cut)
            if include_boundary
            else pred.sum() * 0.0
        )
        return {
            "cls_raw": raw_cls,
            "region_raw": raw_region,
            "boundary_raw": raw_boundary,
            "cls": raw_cls * self.lambda_cls,
            "region": raw_region * self.lambda_region,
            "boundary": raw_boundary * self.lambda_boundary,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        cut: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> torch.Tensor:
        parts = self.compute(pred, target, valid, weight, cut, include_boundary)
        return parts["cls"] + parts["region"] + parts["boundary"]

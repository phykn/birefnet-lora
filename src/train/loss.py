import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.lora.model import ModelOutput


def _resize_targets(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.clamp(0, 1)
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    else:
        valid_mask = valid_mask.clamp(0, 1)
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, size=pred.shape[2:], mode="nearest")
        valid_mask = F.interpolate(valid_mask, size=pred.shape[2:], mode="nearest")
    return target, valid_mask


def _average_masked(value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    denom = valid_mask.sum().clamp_min(1.0)
    return (value * valid_mask).sum() / denom


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
        valid_mask: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        if weight is None:
            weight = torch.ones_like(target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = -torch.expm1(-self.q * bce) / self.q
        return _average_masked(loss * weight, valid_mask)


class IoULoss(nn.Module):
    """Sample-wise soft IoU with an explicit empty-empty=perfect contract."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target * valid_mask).sum(dim=dims)
        union = ((pred + target - pred * target) * valid_mask).sum(dim=dims)
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
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target * valid_mask).sum(dim=dims)
        total = ((pred + target) * valid_mask).sum(dim=dims)
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
        valid_mask: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = make_band(target, self.radius) * valid_mask
        if active.sum() <= 0:
            return pred.sum() * 0.0
        if weight is None:
            weight = torch.ones_like(target)
        pixel_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return _average_masked(pixel_bce * weight, active)


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
        valid_mask: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> dict[str, torch.Tensor]:
        target, valid_mask = _resize_targets(pred, target, valid_mask)
        if weight is None:
            weight = torch.ones_like(target)
        elif weight.shape[2:] != pred.shape[2:]:
            weight = F.interpolate(weight, size=pred.shape[2:], mode="area")
        weight = weight.clamp(0, 1)
        raw_cls = self.cls(pred, target, valid_mask, weight)
        raw_region = self.region(pred.sigmoid(), target, valid_mask * weight)
        raw_boundary = (
            self.boundary(pred, target, valid_mask, weight)
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
        valid_mask: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> torch.Tensor:
        parts = self.compute(pred, target, valid_mask, weight, include_boundary)
        return parts["cls"] + parts["region"] + parts["boundary"]


class TrainLoss(nn.Module):
    def __init__(
        self,
        gce_q: float = 0.7,
        lambda_cls: float = 1.0,
        lambda_region: float = 1.0,
        lambda_boundary: float = 0.5,
        region_loss: str = "dice",
        boundary_radius: int = 3,
        lambda_aux: float = 1.0,
        teacher_confidence: float = 0.95,
        min_gt_weight: float = 0.25,
        lambda_teacher: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.5 <= teacher_confidence < 1.0:
            raise ValueError("teacher confidence must be in [0.5, 1)")
        if not 0.0 <= min_gt_weight <= 1.0:
            raise ValueError("minimum GT weight must be in [0, 1]")
        self.seg = SegmentationLoss(
            gce_q=gce_q,
            lambda_cls=lambda_cls,
            lambda_region=lambda_region,
            lambda_boundary=lambda_boundary,
            region_loss=region_loss,
            boundary_radius=boundary_radius,
        )
        self.lambda_aux = float(lambda_aux)
        self.teacher_confidence = float(teacher_confidence)
        self.min_gt_weight = float(min_gt_weight)
        self.lambda_teacher = float(lambda_teacher)

    def _compute_segments(
        self,
        preds: list[torch.Tensor],
        masks: torch.Tensor,
        valid_masks: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for index, pred in enumerate(preds):
            parts = self.seg.compute(
                pred,
                masks,
                valid_masks,
                weights,
                include_boundary=index == len(preds) - 1,
            )
            for key, value in parts.items():
                weight = 1.0 if key.startswith("boundary") else 1.0 / len(preds)
                totals[key] = totals.get(key, value * 0.0) + value * weight
        totals["seg"] = totals["cls"] + totals["region"] + totals["boundary"]
        return totals

    def _make_weight(
        self,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher_logits.shape[2:] != target.shape[2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        probability = teacher_logits.detach().sigmoid()
        confidence = torch.maximum(probability, 1.0 - probability)
        confidence = (
            (confidence - self.teacher_confidence)
            / (1.0 - self.teacher_confidence)
        ).clamp(0, 1)
        disagree = (probability >= 0.5) != (target >= 0.5)
        conflict = confidence * disagree.to(confidence.dtype) * scale
        weight = 1.0 - (1.0 - self.min_gt_weight) * conflict
        return weight, confidence, probability

    @staticmethod
    def _compute_teacher(
        student_logits: torch.Tensor,
        probability: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape[2:] != probability.shape[2:]:
            probability = F.interpolate(
                probability,
                size=student_logits.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            confidence = F.interpolate(
                confidence,
                size=student_logits.shape[2:],
                mode="area",
            )
        if valid_mask.shape[2:] != student_logits.shape[2:]:
            valid_mask = F.interpolate(
                valid_mask,
                size=student_logits.shape[2:],
                mode="nearest",
            )
        loss = F.binary_cross_entropy_with_logits(
            student_logits,
            probability,
            reduction="none",
        )
        return _average_masked(loss * confidence, valid_mask)

    @staticmethod
    def _compute_gdt(
        gdt: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if gdt is None:
            raise RuntimeError("Training output is missing GDT predictions")
        preds, labels = gdt
        if not preds or len(preds) != len(labels):
            raise RuntimeError("GDT predictions and labels do not match")

        loss = valid_mask.new_zeros(())
        for pred, label in zip(preds, labels):
            if pred.shape[2:] != label.shape[2:]:
                pred = F.interpolate(
                    pred,
                    size=label.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            mask = valid_mask
            if mask.shape[2:] != label.shape[2:]:
                mask = F.interpolate(mask, size=label.shape[2:], mode="nearest")
            pixel_bce = F.binary_cross_entropy_with_logits(
                pred,
                label.sigmoid(),
                reduction="none",
            )
            loss = loss + _average_masked(pixel_bce, mask)
        return loss / len(preds)

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        teacher_logits: torch.Tensor | None = None,
        teacher_scale: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        image_1 = batch["image_1"].to(device)
        mask = batch["mask"].to(device)
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones_like(mask)
        else:
            valid_mask = valid_mask.to(device)

        if model.training:
            image_2 = batch["image_2"].to(device)
            batch_size = image_1.shape[0]
            images = torch.cat([image_1, image_2], dim=0)
            masks = torch.cat([mask, mask], dim=0)
            valid_masks = torch.cat([valid_mask, valid_mask], dim=0)
            out: ModelOutput = model(images)
            preds = out.preds

            logits_1 = preds[-1][:batch_size]
            logits_2 = preds[-1][batch_size:]
            zero = logits_1.new_zeros(())

            if teacher_logits is None or teacher_scale <= 0.0:
                gt_weight = torch.ones_like(mask)
                teacher_raw = zero
            else:
                gt_weight, confidence, probability = self._make_weight(
                    teacher_logits,
                    mask,
                    teacher_scale,
                )
                teacher_raw = self._compute_teacher(
                    logits_2,
                    probability,
                    confidence,
                    valid_mask,
                )
            weights = torch.cat([gt_weight, gt_weight], dim=0)
            loss_dict = self._compute_segments(
                preds,
                masks,
                valid_masks,
                weights,
            )
            aux_raw = (
                self._compute_gdt(out.gdt, valid_masks)
                if self.lambda_aux
                else zero
            )
            teacher_loss = self.lambda_teacher * teacher_scale * teacher_raw
            aux_loss = self.lambda_aux * aux_raw

            loss = loss_dict["seg"] + teacher_loss + aux_loss
            loss_dict.update(
                {
                    "loss": loss,
                    "gt_weight": _average_masked(gt_weight, valid_mask),
                    "teacher_raw": teacher_raw,
                    "teacher": teacher_loss,
                    "aux_raw": aux_raw,
                    "aux": aux_loss,
                }
            )
            return loss_dict, loss

        out = model(image_1)
        loss_dict = self._compute_segments(out.preds[-1:], mask, valid_mask)
        return loss_dict, loss_dict["seg"]

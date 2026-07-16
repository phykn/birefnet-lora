import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapt.wrap import Output


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


def _mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denom = valid.sum().clamp_min(1.0)
    return (value * valid).sum() / denom


def _erode(valid: torch.Tensor, radius: int) -> torch.Tensor:
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
        return _mean(loss * weight, valid)


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
    ) -> torch.Tensor:
        active = make_band(target, self.radius) * valid
        if active.sum() <= 0:
            return pred.sum() * 0.0
        if weight is None:
            weight = torch.ones_like(target)
        pixel_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return _mean(pixel_bce * weight, active)


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
            self.boundary(pred, target, valid, weight)
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
        include_boundary: bool = True,
    ) -> torch.Tensor:
        parts = self.compute(pred, target, valid, weight, include_boundary)
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

    def _segment(
        self,
        logits: list[torch.Tensor],
        target: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for index, pred in enumerate(logits):
            parts = self.seg.compute(
                pred,
                target,
                valid,
                weight,
                include_boundary=index == len(logits) - 1,
            )
            for key, value in parts.items():
                scale = 1.0 if key.startswith("boundary") else 1.0 / len(logits)
                totals[key] = totals.get(key, value * 0.0) + value * scale
        totals["seg"] = totals["cls"] + totals["region"] + totals["boundary"]
        return totals

    def _weigh(
        self,
        teacher: torch.Tensor,
        target: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher.shape[2:] != target.shape[2:]:
            teacher = F.interpolate(
                teacher,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        prob = teacher.detach().sigmoid()
        conf = torch.maximum(prob, 1.0 - prob)
        conf = (
            (conf - self.teacher_confidence)
            / (1.0 - self.teacher_confidence)
        ).clamp(0, 1)
        disagree = (prob >= 0.5) != (target >= 0.5)
        conflict = conf * disagree.to(conf.dtype) * scale
        weight = 1.0 - (1.0 - self.min_gt_weight) * conflict
        return weight, conf, prob

    @staticmethod
    def _distill(
        student: torch.Tensor,
        prob: torch.Tensor,
        conf: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if student.shape[2:] != prob.shape[2:]:
            prob = F.interpolate(
                prob,
                size=student.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            conf = F.interpolate(
                conf,
                size=student.shape[2:],
                mode="area",
            )
        if valid.shape[2:] != student.shape[2:]:
            valid = F.interpolate(
                valid,
                size=student.shape[2:],
                mode="nearest",
            )
        loss = F.binary_cross_entropy_with_logits(
            student,
            prob,
            reduction="none",
        )
        return _mean(loss * conf, valid)

    @staticmethod
    def _guide(
        gdt: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if gdt is None:
            raise RuntimeError("Training output is missing GDT predictions")
        preds, labels = gdt
        if not preds or len(preds) != len(labels):
            raise RuntimeError("GDT predictions and labels do not match")

        loss = valid.new_zeros(())
        for pred, label in zip(preds, labels):
            if pred.shape[2:] != label.shape[2:]:
                pred = F.interpolate(
                    pred,
                    size=label.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            mask = valid
            if mask.shape[2:] != label.shape[2:]:
                mask = F.interpolate(mask, size=label.shape[2:], mode="nearest")
            # The GDT target uses a 5x5 Laplacian, so its padding border is artificial.
            mask = _erode(mask, radius=2)
            pixel_bce = F.binary_cross_entropy_with_logits(
                pred,
                label.detach().sigmoid(),
                reduction="none",
            )
            loss = loss + _mean(pixel_bce, mask)
        return loss / len(preds)

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        teacher_logit: torch.Tensor | None = None,
        teacher_scale: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        device = next(model.parameters()).device
        weak = batch["weak"].to(device)
        mask = batch["mask"].to(device)
        valid = batch.get("valid")
        if valid is None:
            valid = torch.ones_like(mask)
        else:
            valid = valid.to(device)

        if model.training:
            strong = batch["strong"].to(device)
            size = weak.shape[0]
            inputs = torch.cat([weak, strong], dim=0)
            targets = torch.cat([mask, mask], dim=0)
            valids = torch.cat([valid, valid], dim=0)
            out: Output = model(inputs)
            logits = out.logits

            weak_logit = logits[-1][:size]
            strong_logit = logits[-1][size:]
            zero = weak_logit.new_zeros(())

            if teacher_logit is None or teacher_scale <= 0.0:
                gt_weight = torch.ones_like(mask)
                teacher_raw = zero
            else:
                gt_weight, conf, prob = self._weigh(
                    teacher_logit,
                    mask,
                    teacher_scale,
                )
                teacher_raw = self._distill(
                    strong_logit,
                    prob,
                    conf,
                    valid,
                )
            weight = torch.cat([gt_weight, gt_weight], dim=0)
            parts = self._segment(
                logits,
                targets,
                valids,
                weight,
            )
            aux_raw = (
                self._guide(out.gdt, valids)
                if self.lambda_aux
                else zero
            )
            teacher_loss = self.lambda_teacher * teacher_scale * teacher_raw
            aux_loss = self.lambda_aux * aux_raw

            loss = parts["seg"] + teacher_loss + aux_loss
            parts.update(
                {
                    "loss": loss,
                    "gt_weight": _mean(gt_weight, valid),
                    "teacher_raw": teacher_raw,
                    "teacher": teacher_loss,
                    "aux_raw": aux_raw,
                    "aux": aux_loss,
                }
            )
            return parts, loss

        out = model(weak)
        parts = self._segment(out.logits[-1:], mask, valid)
        return parts, parts["seg"]

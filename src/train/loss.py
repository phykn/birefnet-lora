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
        pred = pred * valid_mask
        target = target * valid_mask
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
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
        pred = pred * valid_mask
        target = target * valid_mask
        dims = tuple(range(1, pred.ndim))
        intersection = (pred * target).sum(dim=dims)
        total = pred.sum(dim=dims) + target.sum(dim=dims)
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
    ) -> torch.Tensor:
        active = make_band(target, self.radius) * valid_mask
        if active.sum() <= 0:
            return pred.sum() * 0.0
        pixel_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return _average_masked(pixel_bce, active)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_region: float = 1.0,
        lambda_boundary: float = 1.0,
        region_loss: str = "dice",
        boundary_radius: int = 3,
    ) -> None:
        super().__init__()
        if region_loss not in {"dice", "iou"}:
            raise ValueError("region_loss must be 'dice' or 'iou'")
        self.region = DiceLoss() if region_loss == "dice" else IoULoss()
        self.boundary = BoundaryBCELoss(radius=boundary_radius)
        self.lambda_bce = float(lambda_bce)
        self.lambda_region = float(lambda_region)
        self.lambda_boundary = float(lambda_boundary)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> dict[str, torch.Tensor]:
        target, valid_mask = _resize_targets(pred, target, valid_mask)
        pixel_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        raw_bce = _average_masked(pixel_bce, valid_mask)
        raw_region = self.region(pred.sigmoid(), target, valid_mask)
        raw_boundary = (
            self.boundary(pred, target, valid_mask)
            if include_boundary
            else pred.sum() * 0.0
        )
        return {
            "bce_raw": raw_bce,
            "region_raw": raw_region,
            "boundary_raw": raw_boundary,
            "bce": raw_bce * self.lambda_bce,
            "region": raw_region * self.lambda_region,
            "boundary": raw_boundary * self.lambda_boundary,
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        include_boundary: bool = True,
    ) -> torch.Tensor:
        parts = self.compute(pred, target, valid_mask, include_boundary)
        return parts["bce"] + parts["region"] + parts["boundary"]


class SymmetricBinaryKLLoss(nn.Module):
    @staticmethod
    def _binary_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(p_logits)
        log_p, log_1mp = F.logsigmoid(p_logits), F.logsigmoid(-p_logits)
        log_q, log_1mq = F.logsigmoid(q_logits), F.logsigmoid(-q_logits)
        return p * (log_p - log_q) + (1 - p) * (log_1mp - log_1mq)

    def forward(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits_1.shape != logits_2.shape:
            raise AssertionError(
                "SymmetricBinaryKLLoss expects matching shapes, got "
                f"{tuple(logits_1.shape)} vs {tuple(logits_2.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(logits_1)
        elif valid_mask.shape[2:] != logits_1.shape[2:]:
            valid_mask = F.interpolate(
                valid_mask, size=logits_1.shape[2:], mode="nearest"
            )
        kl = self._binary_kl(logits_1, logits_2) + self._binary_kl(logits_2, logits_1)
        return 0.5 * _average_masked(kl, valid_mask)


class AreaConsistencyLoss(nn.Module):
    def forward(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits_1.shape != logits_2.shape:
            raise AssertionError("AreaConsistencyLoss expects matching shapes")
        if valid_mask is None:
            valid_mask = torch.ones_like(logits_1)
        elif valid_mask.shape[2:] != logits_1.shape[2:]:
            valid_mask = F.interpolate(
                valid_mask, size=logits_1.shape[2:], mode="nearest"
            )
        dims = tuple(range(1, logits_1.ndim))
        denom = valid_mask.sum(dim=dims).clamp_min(1.0)
        area_1 = (torch.sigmoid(logits_1) * valid_mask).sum(dim=dims) / denom
        area_2 = (torch.sigmoid(logits_2) * valid_mask).sum(dim=dims) / denom
        return (area_1 - area_2).abs().mean()


class TrainLoss(nn.Module):
    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_region: float = 1.0,
        lambda_boundary: float = 1.0,
        region_loss: str = "dice",
        boundary_radius: int = 3,
        lambda_kl: float = 0.0,
        lambda_aux: float = 1.0,
        lambda_area: float = 0.0,
    ) -> None:
        super().__init__()
        self.seg = SegmentationLoss(
            lambda_bce=lambda_bce,
            lambda_region=lambda_region,
            lambda_boundary=lambda_boundary,
            region_loss=region_loss,
            boundary_radius=boundary_radius,
        )
        self.con = SymmetricBinaryKLLoss()
        self.area = AreaConsistencyLoss()
        self.lambda_kl = float(lambda_kl)
        self.lambda_area = float(lambda_area)
        self.lambda_aux = float(lambda_aux)

    def _compute_segments(
        self,
        preds: list[torch.Tensor],
        masks: torch.Tensor,
        valid_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for index, pred in enumerate(preds):
            parts = self.seg.compute(
                pred,
                masks,
                valid_masks,
                include_boundary=index == len(preds) - 1,
            )
            for key, value in parts.items():
                weight = 1.0 if key.startswith("boundary") else 1.0 / len(preds)
                totals[key] = totals.get(key, value * 0.0) + value * weight
        totals["seg"] = totals["bce"] + totals["region"] + totals["boundary"]
        return totals

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
        self, model: nn.Module, batch: dict[str, torch.Tensor]
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

            loss_dict = self._compute_segments(preds, masks, valid_masks)
            zero = logits_1.new_zeros(())
            con_raw = (
                self.con(logits_1, logits_2, valid_mask)
                if self.lambda_kl
                else zero
            )
            area_raw = (
                self.area(logits_1, logits_2, valid_mask)
                if self.lambda_area
                else zero
            )
            aux_raw = (
                self._compute_gdt(out.gdt, valid_masks)
                if self.lambda_aux
                else zero
            )
            con_loss = self.lambda_kl * con_raw
            area_loss = self.lambda_area * area_raw
            aux_loss = self.lambda_aux * aux_raw

            loss = loss_dict["seg"] + con_loss + area_loss + aux_loss
            loss_dict.update(
                {
                    "loss": loss,
                    "con_raw": con_raw,
                    "con": con_loss,
                    "area_raw": area_raw,
                    "area": area_loss,
                    "aux_raw": aux_raw,
                    "aux": aux_loss,
                }
            )
            return loss_dict, loss

        out = model(image_1)
        loss_dict = self._compute_segments(out.preds[-1:], mask, valid_mask)
        return loss_dict, loss_dict["seg"]

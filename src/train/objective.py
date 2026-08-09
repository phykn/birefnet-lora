import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.output import Output
from .losses import SegmentationLoss, erode_valid, masked_mean


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
        cut: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for index, pred in enumerate(logits):
            parts = self.seg.compute(
                pred,
                target,
                valid,
                weight,
                cut,
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
        return masked_mean(loss * conf, valid)

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
            mask = erode_valid(mask, radius=2)
            pixel_bce = F.binary_cross_entropy_with_logits(
                pred,
                label.detach().sigmoid(),
                reduction="none",
            )
            loss = loss + masked_mean(pixel_bce, mask)
        return loss / len(preds)

    def forward(
        self,
        out: Output,
        batch: dict[str, torch.Tensor],
        teacher_logit: torch.Tensor | None = None,
        teacher_scale: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        mask = batch["mask"]
        valid = batch.get("valid")
        if valid is None:
            valid = torch.ones_like(mask)
        cut = batch.get("cut")
        if cut is None:
            cut = torch.zeros_like(mask)

        if out.gdt is not None:
            size = mask.shape[0]
            targets = torch.cat([mask, mask], dim=0)
            valids = torch.cat([valid, valid], dim=0)
            cuts = torch.cat([cut, cut], dim=0)
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
                cuts,
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
                    "gt_weight": masked_mean(gt_weight, valid),
                    "teacher_raw": teacher_raw,
                    "teacher": teacher_loss,
                    "aux_raw": aux_raw,
                    "aux": aux_loss,
                }
            )
            return parts, loss

        parts = self._segment(out.logits[-1:], mask, valid, cut=cut)
        return parts, parts["seg"]

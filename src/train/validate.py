import numpy as np
import torch
import torch.nn.functional as F

from ..predict.run import predict_logits
from ..prepare.read import read_image, read_mask
from .score import (
    boundary,
    boundary_logits,
    brier,
    dice,
    ece,
    iou_logits,
)


def _score_binary(
    pred: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    total = int(pred.sum() + target.sum())
    iou = 1.0 if union == 0 else intersection / union
    dice = 1.0 if total == 0 else 2.0 * intersection / total
    return iou, dice


class ValidationMixin:
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        sample_count = 0

        for batch in self.valid_loader:
            image = batch["weak"].to(self.device)
            target = batch["mask"].to(self.device)
            valid = batch["valid"].to(self.device)
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss_dict, _ = self.criterion(self.model, batch)
                logits = self.model(image).logits[-1]

            if logits.shape[2:] != target.shape[2:]:
                target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
                valid = F.interpolate(
                    valid, size=logits.shape[2:], mode="nearest"
                )
            batch_size = image.shape[0]
            sample_count += batch_size
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + value.item() * batch_size
            totals["region_iou"] = (
                totals.get("region_iou", 0.0)
                + float(iou_logits(logits, target, valid)) * batch_size
            )
            totals["dice"] = (
                totals.get("dice", 0.0)
                + float(dice(logits, target, valid)) * batch_size
            )
            totals["brier"] = (
                totals.get("brier", 0.0)
                + float(brier(logits, target, valid))
                * batch_size
            )
            totals["ece"] = (
                totals.get("ece", 0.0)
                + float(ece(logits, target, valid))
                * batch_size
            )
            totals["boundary_f1_2px"] = (
                totals.get("boundary_f1_2px", 0.0)
                + boundary_logits(
                    logits.float(), target, valid, tolerance_px=2.0
                )
                * batch_size
            )

        if sample_count == 0:
            raise RuntimeError("Validation loader is empty")
        return {key: value / sample_count for key, value in totals.items()}

    def predict_native(self, loader):
        pairs = getattr(loader.dataset, "data", None)
        if not pairs:
            raise RuntimeError("Deployment validation dataset has no image/mask pairs")
        self.model.eval()
        for image_path, mask_path in pairs:
            image = read_image(image_path)
            target = read_mask(mask_path) > 127
            logits = predict_logits(
                self.model,
                image,
                size=self.inference["size"],
                mode=self.inference["mode"],
                overlap_ratio=self.inference["overlap_ratio"],
                tile_batch=self.inference["tile_batch"],
                context_weight=self.inference["context_weight"],
            )
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            yield prob, target

    def calibrate(self) -> float:
        thresholds = [index / 100 for index in range(30, 71)]
        totals = {threshold: 0.0 for threshold in thresholds}
        count = 0
        for prob, target in self.predict_native(self.calib_loader):
            count += 1
            for threshold in thresholds:
                region, _ = _score_binary(prob >= threshold, target)
                totals[threshold] += region
        self.calib_threshold = min(
            thresholds,
            key=lambda threshold: (-totals[threshold] / count, abs(threshold - 0.5)),
        )
        return self.calib_threshold

    def validate_deploy(self, threshold: float) -> dict[str, float]:
        totals = {"region_iou": 0.0, "dice": 0.0, "boundary_f1": 0.0}
        count = 0
        for prob, target in self.predict_native(self.valid_loader):
            count += 1
            pred = prob >= threshold
            region, overlap = _score_binary(pred, target)
            totals["region_iou"] += region
            totals["dice"] += overlap
            totals["boundary_f1"] += boundary(pred, target, tolerance_px=2.0)
        return {f"deploy_{key}": value / count for key, value in totals.items()}

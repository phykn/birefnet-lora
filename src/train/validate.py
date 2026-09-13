from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.image import read_image, read_mask
from ..data.pairs import Pair
from ..prepare.spec import PreprocessSpec
from .metrics import (
    boundary,
    boundary_logits,
    brier,
    dice,
    ece,
    iou_at_thresholds,
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


def _dataset_pairs(dataset: object) -> list[Pair]:
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        # Compatibility for dataset implementations using the former field.
        pairs = getattr(dataset, "data", None)
    if not pairs:
        raise RuntimeError(
            "Deployment validation dataset has no image/mask pairs"
        )
    return pairs


class Validator:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        valid_loader: DataLoader,
        calib_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        amp_dtype: torch.dtype,
        use_amp: bool,
        preprocess: PreprocessSpec,
        predictor: Callable[..., np.ndarray],
    ) -> None:
        self.model = model
        self.valid_loader = valid_loader
        self.calib_loader = calib_loader
        self.criterion = criterion
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp
        self.preprocess = preprocess
        self.predictor = predictor

    def _move(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        non_blocking = self.device.type == "cuda"
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.model.training:
            self.model.eval()
        totals: dict[str, float] = {}
        sample_count = 0

        for cpu_batch in self.valid_loader:
            batch = self._move(cpu_batch)
            image = batch["weak"]
            target = batch["mask"]
            valid = batch["valid"]
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                out = self.model(image)
                loss_dict, _ = self.criterion(out, batch)
                logits = out.logits[-1]

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
        pairs = _dataset_pairs(loader.dataset)
        if self.model.training:
            self.model.eval()
        for image_path, mask_path in pairs:
            image = read_image(image_path)
            target = read_mask(mask_path) > 127
            logits = self.predictor(
                self.model,
                image,
                size=self.preprocess.size,
                mode=self.preprocess.mode,
            )
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            yield prob, target

    def calibrate(self) -> float:
        thresholds = tuple(index / 100 for index in range(30, 71))
        totals = np.zeros(len(thresholds), dtype=np.float64)
        count = 0
        for prob, target in self.predict_native(self.calib_loader):
            count += 1
            totals += iou_at_thresholds(prob, target, thresholds)
        if count == 0:
            raise RuntimeError("Calibration loader is empty")
        scores = totals / count
        best = min(
            range(len(thresholds)),
            key=lambda index: (-scores[index], abs(thresholds[index] - 0.5)),
        )
        return thresholds[best]

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

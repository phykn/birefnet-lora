from pathlib import Path
from typing import Any

import torch

from ..prepare.spec import PreprocessSpec
from ..storage import atomic_torch_save


class CheckpointStore:
    def __init__(
        self,
        *,
        model: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        teacher: Any,
        save_dir: str,
        preprocess: PreprocessSpec,
        global_step: int,
        best_region: float,
        best_boundary: float,
        calib_threshold: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.teacher = teacher
        self.save_dir = save_dir
        self.preprocess = preprocess
        self.global_step = global_step
        self.best_region = best_region
        self.best_boundary = best_boundary
        self.calib_threshold = calib_threshold

    def _overlay_extra(
        self,
        extra: dict | None = None,
    ) -> dict:
        value = {"preprocess": self.preprocess.to_meta()}
        if extra:
            value.update(extra)
        return value

    def save(self) -> None:
        weights_dir = Path(self.save_dir) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        overlay = self.model.make_overlay(self._overlay_extra())
        atomic_torch_save(overlay, weights_dir / "last.overlay.pth")
        training_state = {
            "overlay": overlay,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "teacher": self.teacher.state_dict(),
            "global_step": self.global_step,
            "best_region": self.best_region,
            "best_boundary": self.best_boundary,
            "threshold": self.calib_threshold,
        }
        atomic_torch_save(training_state, weights_dir / "last.train.pth")

    def save_best(self, name: str, metrics: dict[str, float]) -> None:
        weights_dir = Path(self.save_dir) / "weights"
        extra = self._overlay_extra(
            {
                "selection": {
                    "name": name,
                    "global_step": self.global_step,
                    "metrics": metrics,
                    "threshold": self.calib_threshold,
                },
            }
        )
        self.model.save_overlay(
            str(weights_dir / f"best_{name}.overlay.pth"),
            extra=extra,
        )

    def load_resume(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "overlay" not in state:
            raise RuntimeError("Unsupported training checkpoint format")
        self.model.load_payload(state["overlay"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.teacher.load_state_dict(state["teacher"])
        self.global_step = int(state["global_step"])
        self.best_region = float(state["best_region"])
        self.best_boundary = float(state["best_boundary"])
        self.calib_threshold = float(state["threshold"])


class CheckpointMixin:
    """Compatibility mixin for callers outside the composed Trainer."""

    def _checkpoint_store(self) -> CheckpointStore:
        return CheckpointStore(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            teacher=self.teacher,
            save_dir=self.save_dir,
            preprocess=self.preprocess,
            global_step=self.global_step,
            best_region=self.best_region,
            best_boundary=self.best_boundary,
            calib_threshold=self.calib_threshold,
        )

    def _overlay_extra(self, extra: dict | None = None) -> dict:
        return self._checkpoint_store()._overlay_extra(extra)

    def save(self) -> None:
        self._checkpoint_store().save()

    def save_best(self, name: str, metrics: dict[str, float]) -> None:
        self._checkpoint_store().save_best(name, metrics)

    def load_resume(self, path: str) -> None:
        store = self._checkpoint_store()
        store.load_resume(path)
        self.global_step = store.global_step
        self.best_region = store.best_region
        self.best_boundary = store.best_boundary
        self.calib_threshold = store.calib_threshold
        self._train_iter = None

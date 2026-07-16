import os
import uuid
from pathlib import Path
from typing import Any

import torch


def _save(payload: dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


class CheckpointMixin:
    def save(self) -> None:
        weights_dir = Path(self.save_dir) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        overlay = self.model.make_overlay({"inference": self.inference})
        _save(overlay, str(weights_dir / "last.overlay.pth"))
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
        _save(training_state, str(weights_dir / "last.train.pth"))

    def save_best(self, name: str, metrics: dict[str, float]) -> None:
        weights_dir = Path(self.save_dir) / "weights"
        extra = {
            "inference": self.inference,
            "selection": {
                "name": name,
                "global_step": self.global_step,
                "metrics": metrics,
                "threshold": self.calib_threshold,
            },
        }
        self.model.save_overlay(
            str(weights_dir / f"best_{name}.overlay.pth"),
            extra=extra,
        )

    def load_resume(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "overlay" not in state:
            raise RuntimeError("Unsupported training checkpoint format")
        meta = self.model.load_payload(state["overlay"])
        if meta.get("inference") != self.inference:
            raise RuntimeError("Resume checkpoint inference config does not match")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.teacher.load_state_dict(state["teacher"])
        self.global_step = int(state["global_step"])
        self.best_region = float(state["best_region"])
        self.best_boundary = float(state["best_boundary"])
        self.calib_threshold = float(state["threshold"])
        self._train_iter = None

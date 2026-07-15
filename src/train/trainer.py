import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .schedule import CosineSchedule
from .state import CheckpointMixin
from .teacher import Teacher
from .validate import ValidationMixin


class Trainer(ValidationMixin, CheckpointMixin):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        calib_loader: DataLoader,
        inference: dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineSchedule,
        teacher: Teacher,
        save_dir: str,
        max_grad_norm: float = 1.0,
        accum_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.calib_loader = calib_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher = teacher
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.accum_steps = accum_steps
        self.inference = inference
        self.device = next(model.parameters()).device
        self.global_step = 0
        self.best_region = float("-inf")
        self.best_boundary = float("-inf")
        self.calib_threshold = 0.5

        self.use_amp = self.device.type == "cuda"
        if self.use_amp and torch.cuda.is_bf16_supported(including_emulation=False):
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self._writer: SummaryWriter | None = None
        self._train_iter = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum: dict[str, float] = {}
        teacher_scale = self.teacher.scale(self.global_step + 1)
        for _ in range(self.accum_steps):
            batch = self.next_batch()
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                teacher_logits = None
                if teacher_scale > 0.0:
                    image = batch["image_1"].to(self.device)
                    teacher_logits = self.teacher.predict(self.model, image)
                loss_dict, loss = self.criterion(
                    self.model,
                    batch,
                    teacher_logits=teacher_logits,
                    teacher_scale=teacher_scale,
                )
            self.scaler.scale(loss / self.accum_steps).backward()
            for key, value in loss_dict.items():
                accum[key] = accum.get(key, 0.0) + value.item() / self.accum_steps

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.list_trainable(), self.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.teacher.update(self.model)
        self.scheduler.step()
        self.global_step += 1
        accum["grad_norm"] = float(grad_norm)
        return accum

    def train(
        self,
        steps: int,
        val_freq: int = 500,
        save_freq: int = 1000,
    ) -> None:
        self._writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "logs"))
        progress_bar = tqdm(range(self.global_step + 1, steps + 1), desc="Training")

        for _ in progress_bar:
            losses = self.step()
            progress_bar.set_postfix(
                {key: f"{value:.4f}" for key, value in losses.items()}
            )
            for key, value in losses.items():
                self._writer.add_scalar(f"train/{key}", value, self.global_step)
            for index, group in enumerate(self.optimizer.param_groups):
                name = group.get("name", str(index))
                self._writer.add_scalar(f"lr/{name}", group["lr"], self.global_step)

            if self.global_step % val_freq == 0:
                valid_metrics = self.validate()
                valid_metrics.update(self.validate_deploy(0.5))
                for key, value in valid_metrics.items():
                    self._writer.add_scalar(f"valid/{key}", value, self.global_step)
                region_improved = valid_metrics["deploy_region_iou"] > self.best_region
                boundary_improved = (
                    valid_metrics["deploy_boundary_f1"] > self.best_boundary
                )
                if region_improved or boundary_improved:
                    threshold = self.calibrate()
                    self._writer.add_scalar(
                        "valid/threshold", threshold, self.global_step
                    )
                if region_improved:
                    self.best_region = valid_metrics["deploy_region_iou"]
                    self.save_best("region", valid_metrics)
                if boundary_improved:
                    self.best_boundary = valid_metrics["deploy_boundary_f1"]
                    self.save_best("boundary", valid_metrics)

            if self.global_step % save_freq == 0:
                self.save()

        self.save()
        self._writer.flush()
        self._writer.close()

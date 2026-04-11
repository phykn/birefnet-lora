import os

import torch
import torch.nn as nn
from .scheduler import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingWarmupRestarts,
        save_dir: str,
        max_grad_norm: float = 1.0,
        accum_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.accum_steps = accum_steps
        self.device = next(model.parameters()).device

        self.use_amp = self.device.type == "cuda"
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        # GradScaler is only needed for fp16; bf16 has fp32-equivalent range.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        self._writer: SummaryWriter | None = None
        self._train_iter = None

    def get_batch(self) -> dict[str, torch.Tensor]:
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
        for _ in range(self.accum_steps):
            batch = self.get_batch()
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss_dict, loss = self.criterion(self.model, batch)
            self.scaler.scale(loss / self.accum_steps).backward()
            for k, v in loss_dict.items():
                accum[k] = accum.get(k, 0.0) + v.item() / self.accum_steps

        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.get_adapter_params(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return accum

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}

        for batch in self.valid_loader:
            with torch.amp.autocast(
                self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                loss_dict, _ = self.criterion(self.model, batch)
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + v.item()

        num_batches = len(self.valid_loader)
        return {k: v / num_batches for k, v in totals.items()}

    def save(self) -> None:
        weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        self.model.save_adapters(os.path.join(weights_dir, "last.pth"))

    def train(
        self,
        steps: int,
        val_freq: int = 500,
        save_freq: int = 1000,
    ) -> None:
        log_dir = os.path.join(self.save_dir, "logs")
        self._writer = SummaryWriter(log_dir=log_dir)
        progress_bar = tqdm(range(1, steps + 1), desc="Training")

        for global_step in progress_bar:
            losses = self.step()
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

            for k, v in losses.items():
                self._writer.add_scalar(f"Train/{k.capitalize()}", v, global_step)
            self._writer.add_scalar(
                "Train/LR", self.scheduler.get_last_lr()[0], global_step
            )

            if global_step % val_freq == 0:
                valid_losses = self.validate()
                for k, v in valid_losses.items():
                    self._writer.add_scalar(f"Val/{k.capitalize()}", v, global_step)

            if global_step % save_freq == 0:
                self.save()

        self.save()
        self._writer.flush()
        self._writer.close()

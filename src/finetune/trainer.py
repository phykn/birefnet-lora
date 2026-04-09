import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
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
        cons_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_cons: float = 0.1,
        max_grad_norm: float = 1.0,
        scheduler_name: str = "cosine",
        warmup_steps: int = 50,
        total_steps: int = 1000,
        use_tensorboard: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.cons_criterion = cons_criterion
        self.optimizer = optimizer
        self.device = device
        self.lambda_cons = lambda_cons
        self.max_grad_norm = max_grad_norm

        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)

        self.scheduler = self._build_scheduler(scheduler_name, warmup_steps, total_steps)

        self.best_val_loss = float("inf")

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("run", run_timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        log_dir = os.path.join(self.save_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
        self.train_iter = iter(self.train_loader)

    def _build_scheduler(
        self,
        name: str,
        warmup_steps: int,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if name == "none":
            return None

        warmup = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup_steps, 1)))
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1))
        return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    def save_checkpoint(self, path: str, step: int) -> None:
        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if self.scheduler:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        return state["step"]

    def train(self, steps: int, val_freq: int = 500, save_freq: int = 1000, start_step: int = 0) -> None:
        progress_bar = tqdm(range(start_step + 1, steps + 1), desc="Training")

        for step in progress_bar:
            losses = self._step()
            progress_bar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                seg=f"{losses['seg']:.4f}",
                cons=f"{losses['cons']:.4f}",
            )

            if self.writer:
                self.writer.add_scalar("Train/Loss", losses["loss"], step)
                self.writer.add_scalar("Train/Seg", losses["seg"], step)
                self.writer.add_scalar("Train/Aux", losses["aux"], step)
                self.writer.add_scalar("Train/Cons", losses["cons"], step)
                if self.scheduler:
                    self.writer.add_scalar("Train/LR", self.scheduler.get_last_lr()[0], step)

            if step % val_freq == 0:
                valid_loss = self._validate()
                if self.writer:
                    self.writer.add_scalar("Val/Loss", valid_loss, step)
                if valid_loss < self.best_val_loss:
                    self.best_val_loss = valid_loss
                    self._save(filename="best.pth", step=step)

            if step % save_freq == 0:
                self._save(filename="last.pth", step=step)

        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _step(self) -> dict[str, float]:
        self.model.train()
        batch = self._get_batch()
        images_v1 = batch["image_v1"].to(self.device)
        images_v2 = batch["image_v2"].to(self.device)
        masks = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
            pred1, aux1 = self.model(images_v1)
            pred2, aux2 = self.model(images_v2)

            seg_loss1 = self.criterion(pred1, masks)
            seg_loss2 = self.criterion(pred2, masks)
            seg_loss = seg_loss1 + seg_loss2

            cons_loss = self.cons_criterion(pred1, pred2) * self.lambda_cons
            aux_loss = aux1 + aux2

            total_loss = seg_loss + aux_loss + cons_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "seg": seg_loss.item(),
            "aux": aux_loss.item(),
            "cons": cons_loss.item(),
        }

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.valid_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
                prediction = self.model(images)
                loss = self.criterion(prediction, masks)

            total_loss += loss.item()

        num_batches = len(self.valid_loader)
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save(self, filename: str = "last.pth", step: int = 0) -> None:
        weights_dir = os.path.join(self.save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        if hasattr(self.model, "save_adapters"):
            self.model.save_adapters(os.path.join(weights_dir, filename))
        self.save_checkpoint(os.path.join(weights_dir, "checkpoint.pth"), step=step)

    def _get_batch(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

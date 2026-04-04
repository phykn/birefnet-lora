import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_tensorboard: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.use_amp)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("run", run_timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        log_dir = os.path.join(self.save_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
        self.train_iter = iter(self.train_loader)

    def train(self, steps: int, val_freq: int = 500, save_freq: int = 1000) -> None:
        progress_bar = tqdm(range(1, steps + 1), desc="Training")

        for step in progress_bar:
            losses = self._step()
            progress_bar.set_postfix(
                loss=f"{losses['loss']:.4f}",
                seg=f"{losses['seg']:.4f}",
                aux=f"{losses['aux']:.4f}",
            )

            if self.writer:
                self.writer.add_scalar("Train/Loss", losses["loss"], step)
                self.writer.add_scalar("Train/Seg", losses["seg"], step)
                self.writer.add_scalar("Train/Aux", losses["aux"], step)

            if step % val_freq == 0:
                valid_loss = self._validate()
                if self.writer:
                    self.writer.add_scalar("Val/Loss", valid_loss, step)

            if step % save_freq == 0:
                self._save()

        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _step(self) -> dict[str, float]:
        self.model.train()
        batch = self._get_batch()
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.amp_device_type, enabled=self.use_amp):
            prediction, aux_loss = self.model(images)
            segmentation_loss = self.criterion(prediction, masks)
            total_loss = segmentation_loss + aux_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": total_loss.item(),
            "seg": segmentation_loss.item(),
            "aux": aux_loss.item(),
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

    def _save(self) -> None:
        if hasattr(self.model, "save_adapters"):
            weights_dir = os.path.join(self.save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            path = os.path.join(weights_dir, "last.pth")
            self.model.save_adapters(path)

    def _get_batch(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

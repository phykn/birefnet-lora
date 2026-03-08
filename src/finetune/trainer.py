import os
import torch

from datetime import datetime
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
        self.scaler = torch.amp.GradScaler("cuda")

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("run", now)
        os.makedirs(self.save_dir, exist_ok=True)

        log_dir = os.path.join(self.save_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
        self.train_iter = iter(self.train_loader)

    def train(
        self,
        steps: int,
        val_freq: int = 500,
        save_freq: int = 1000
    ) -> None:
        pbar = tqdm(range(1, steps + 1), desc="Training")

        for step in pbar:
            losses = self._step()
            pbar.set_postfix(loss=f"{losses['loss']:.4f}", seg=f"{losses['seg']:.4f}", aux=f"{losses['aux']:.4f}")

            if self.writer:
                self.writer.add_scalar("Train/Loss", losses["loss"], step)
                self.writer.add_scalar("Train/Seg", losses["seg"], step)
                self.writer.add_scalar("Train/Aux", losses["aux"], step)

            if step % val_freq == 0:
                val_loss = self._validate()
                if self.writer:
                    self.writer.add_scalar("Val/Loss", val_loss, step)

            if step % save_freq == 0:
                self._save()

    def _step(self) -> dict[str, float]:
        self.model.train()
        batch = self._get_batch()
        imgs = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        self.optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out, loss_aux = self.model(imgs)
            loss_seg = self.criterion(out, masks)
            loss = loss_seg + loss_aux

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": loss.item(),
            "seg": loss_seg.item(),
            "aux": loss_aux.item()
        }

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.valid_loader:
            imgs = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.amp.autocast("cuda"):
                out = self.model(imgs)
                loss = self.criterion(out, masks)

            total_loss += loss.item()

        n = len(self.valid_loader)
        return total_loss / n if n > 0 else 0.0

    def _save(self) -> None:
        if hasattr(self.model, "save_adapters"):
            wdir = os.path.join(self.save_dir, "weights")
            os.makedirs(wdir, exist_ok=True)
            path = os.path.join(wdir, "last.pth")
            self.model.save_adapters(path)

    def _get_batch(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

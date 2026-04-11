import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ml.training.scheduler import CosineAnnealingWarmupRestarts
from src.ml.training.trainer import Trainer


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        pred = self.conv(x)
        if self.training:
            return [pred, pred], torch.tensor(0.0, device=x.device)
        return pred

    def get_adapter_params(self):
        return list(self.parameters())

    def save_adapters(self, path):
        torch.save(self.state_dict(), path)


class _DummyCriterion(nn.Module):
    def forward(self, model, batch):
        device = next(model.parameters()).device
        if model.training:
            preds, aux = model(batch["image_1"].to(device))
            seg = sum(p.mean() ** 2 for p in preds) / len(preds)
            cons = (preds[0] - preds[-1]).pow(2).mean()
            loss = seg + cons + aux
            return {"loss": loss, "seg": seg, "cons": cons, "aux": aux}, loss
        pred = model(batch["image_1"].to(device))
        seg = pred.mean() ** 2
        return {"loss": seg, "seg": seg}, seg


class _DummyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image_1": torch.randn(3, 8, 8),
            "mask": torch.randint(0, 2, (1, 8, 8)).float(),
        }


def _make_trainer(tmp_path, monkeypatch, accum_steps=1):
    monkeypatch.chdir(tmp_path)
    model = _DummyModel()
    train_loader = DataLoader(_DummyDataset(4), batch_size=2)
    valid_loader = DataLoader(_DummyDataset(4), batch_size=2)
    criterion = _DummyCriterion()
    optimizer = torch.optim.AdamW(model.get_adapter_params(), lr=1e-2)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=10,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_steps=2,
    )
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=1.0,
        accum_steps=accum_steps,
    )


def test_trainer_step_returns_loss_dict_and_updates_params(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    before = trainer.model.conv.weight.detach().clone()

    losses = trainer.step()

    assert {"loss", "seg", "cons", "aux"} <= set(losses.keys())
    assert all(isinstance(v, float) for v in losses.values())
    after = trainer.model.conv.weight
    assert not torch.allclose(before, after)


def test_trainer_step_with_accum(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, accum_steps=2)
    losses = trainer.step()
    assert "loss" in losses


def test_trainer_validate_returns_avg_losses(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    metrics = trainer.validate()
    assert "loss" in metrics
    assert "seg" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_trainer_get_batch_wraps_around(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    for _ in range(6):
        batch = trainer.get_batch()
        assert "image_1" in batch


def test_trainer_save_writes_adapter_file(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.save()
    weights_path = os.path.join(trainer.save_dir, "weights", "last.pth")
    assert os.path.exists(weights_path)

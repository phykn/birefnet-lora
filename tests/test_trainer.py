import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.adapt.wrap import Output
from src.train.run import Trainer
from src.train.schedule import CosineSchedule
from src.train.teacher import Teacher


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        pred = self.conv(x)
        if self.training:
            return Output(logits=[pred, pred])
        return Output(logits=[pred])

    def list_trainable(self):
        return list(self.parameters())

    def make_overlay(self, extra=None):
        return {
            "meta": extra or {},
            "state": {
                key: value.detach().cpu() for key, value in self.state_dict().items()
            },
        }

    def load_payload(self, payload):
        self.load_state_dict(payload["state"])
        return payload["meta"]

    def save_overlay(self, path, extra=None):
        payload = self.make_overlay(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)


class _DummyCriterion(nn.Module):
    def forward(self, model, batch, teacher_logit=None, teacher_scale=0.0):
        device = next(model.parameters()).device
        if model.training:
            out = model(batch["weak"].to(device))
            logits = out.logits
            seg = sum(logit.mean() ** 2 for logit in logits) / len(logits)
            cons = (logits[0] - logits[-1]).pow(2).mean()
            aux = torch.tensor(0.0, device=device)
            loss = seg + cons + aux
            return {"loss": loss, "seg": seg, "cons": cons, "aux": aux}, loss
        out = model(batch["weak"].to(device))
        pred = out.logits[-1]
        seg = pred.mean() ** 2
        return {"loss": seg, "seg": seg}, seg


class _DummyDataset(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "weak": torch.randn(3, 8, 8),
            "strong": torch.randn(3, 8, 8),
            "mask": torch.randint(0, 2, (1, 8, 8)).float(),
            "valid": torch.ones(1, 8, 8),
        }


def _make_trainer(tmp_path, accum_steps=1):
    model = _DummyModel()
    train_loader = DataLoader(_DummyDataset(4), batch_size=2)
    valid_loader = DataLoader(_DummyDataset(4), batch_size=2)
    criterion = _DummyCriterion()
    optimizer = torch.optim.AdamW(model.list_trainable(), lr=1e-2)
    scheduler = CosineSchedule(
        optimizer=optimizer,
        first_cycle_steps=10,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_steps=2,
    )
    teacher = Teacher(model, decay=0.5, start=0, ramp=1)
    return Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        calib_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher=teacher,
        save_dir=str(tmp_path),
        max_grad_norm=1.0,
        accum_steps=accum_steps,
    )


def test_trainer_step_returns_loss_dict_and_updates_params(tmp_path):
    trainer = _make_trainer(tmp_path)
    before = trainer.model.conv.weight.detach().clone()

    losses, updated = trainer.step()

    assert updated
    assert {"loss", "seg", "cons", "aux"} <= set(losses.keys())
    assert all(isinstance(v, float) for v in losses.values())
    after = trainer.model.conv.weight
    assert not torch.allclose(before, after)


def test_trainer_step_with_accum(tmp_path):
    trainer = _make_trainer(tmp_path, accum_steps=2)
    losses, updated = trainer.step()
    assert updated
    assert "loss" in losses


def test_trainer_validate_returns_avg_losses(tmp_path):
    trainer = _make_trainer(tmp_path)
    metrics = trainer.validate()
    assert "loss" in metrics
    assert "seg" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_trainer_get_batch_wraps_around(tmp_path):
    trainer = _make_trainer(tmp_path)
    for _ in range(6):
        batch = trainer.next_batch()
        assert "weak" in batch


def test_trainer_save_writes_overlay_and_resume_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.save()
    weights_dir = os.path.join(trainer.save_dir, "weights")
    overlay_path = os.path.join(weights_dir, "last.overlay.pth")
    assert os.path.exists(overlay_path)
    assert os.path.exists(os.path.join(weights_dir, "last.train.pth"))
    overlay = torch.load(overlay_path, map_location="cpu", weights_only=True)
    assert "inference" not in overlay["meta"]


def test_trainer_resume_restores_step_model_optimizer_and_scheduler(tmp_path):
    trainer = _make_trainer(tmp_path)
    _, updated = trainer.step()
    assert updated
    expected_weight = trainer.model.conv.weight.detach().clone()
    expected_teacher = trainer.teacher.state_dict()
    expected_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.save()

    resumed = _make_trainer(tmp_path)
    resumed.load_resume(os.path.join(tmp_path, "weights", "last.train.pth"))
    assert resumed.global_step == trainer.global_step
    assert torch.allclose(resumed.model.conv.weight, expected_weight)
    assert resumed.optimizer.param_groups[0]["lr"] == expected_lr
    assert resumed.scheduler.step_in_cycle == trainer.scheduler.step_in_cycle
    for name, value in expected_teacher.items():
        assert torch.allclose(resumed.teacher.state_dict()[name], value)


class _SkipScaler:
    def __init__(self):
        self.value = 2.0
        self.attempts = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.attempts += 1
        if self.attempts > 1:
            optimizer.step()

    def update(self):
        if self.attempts == 1:
            self.value = 1.0

    def get_scale(self):
        return self.value


def test_overflow_skip_keeps_optimizer_dependent_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.scaler = _SkipScaler()
    weight = trainer.model.conv.weight.detach().clone()
    teacher = trainer.teacher.state_dict()
    cycle = trainer.scheduler.step_in_cycle

    _, updated = trainer.step()

    assert not updated
    assert trainer.global_step == 0
    assert trainer.scheduler.step_in_cycle == cycle
    assert torch.allclose(trainer.model.conv.weight, weight)
    for name, value in teacher.items():
        assert torch.allclose(trainer.teacher.state_dict()[name], value)


def test_training_retries_skipped_update(tmp_path):
    trainer = _make_trainer(tmp_path)
    scaler = _SkipScaler()
    trainer.scaler = scaler
    trainer.save = lambda: None

    trainer.train(steps=1, val_freq=2, save_freq=10)

    assert scaler.attempts == 2
    assert trainer.global_step == 1


def test_best_selection_uses_calibrated_deployment_threshold(tmp_path):
    trainer = _make_trainer(tmp_path)
    calls = []

    def calibrate():
        calls.append("calibrate")
        trainer.calib_threshold = 0.63
        return 0.63

    def validate():
        calls.append("validate")
        return {"loss": 0.0}

    def validate_deploy(threshold):
        calls.append(("deploy", threshold))
        return {
            "deploy_region_iou": 0.8,
            "deploy_dice": 0.85,
            "deploy_boundary_f1": 0.7,
        }

    trainer.calibrate = calibrate
    trainer.validate = validate
    trainer.validate_deploy = validate_deploy
    trainer.train(steps=1, val_freq=1, save_freq=10)

    assert calls == ["calibrate", "validate", ("deploy", 0.63)]
    payload = torch.load(
        tmp_path / "weights" / "best_boundary.overlay.pth",
        map_location="cpu",
        weights_only=True,
    )
    assert payload["meta"]["selection"]["threshold"] == 0.63


def test_calibration_and_deployment_validation_use_native_inference(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((8, 12), 255, dtype=np.uint8)).save(mask_path)

    trainer = _make_trainer(tmp_path)
    trainer.valid_loader.dataset.data = [(str(image_path), str(mask_path))]
    with torch.no_grad():
        trainer.model.conv.weight.zero_()
        trainer.model.conv.bias.fill_(20.0)

    threshold = trainer.calibrate()
    metrics = trainer.validate_deploy(threshold)
    assert threshold == 0.5
    assert metrics == {
        "deploy_region_iou": 1.0,
        "deploy_dice": 1.0,
        "deploy_boundary_f1": 1.0,
    }

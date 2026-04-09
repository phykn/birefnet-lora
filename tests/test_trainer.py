import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.finetune.loss import ConsistencyLoss, SegmentationLoss
from src.finetune.trainer import Trainer


def _make_dummy_model():
    """Simple model mimicking LoRABiRefNet interface."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            pred = self.conv(x)
            if self.training:
                aux = torch.tensor(0.0, device=x.device)
                return pred, aux
            return pred

        def get_adapter_params(self):
            return list(self.parameters())

        def save_adapters(self, path):
            torch.save(self.state_dict(), path)

    return DummyModel()


def _make_dual_view_loader(batch_size=2, num_samples=4):
    v1 = torch.randn(num_samples, 3, 32, 32)
    v2 = torch.randn(num_samples, 3, 32, 32)
    masks = torch.rand(num_samples, 1, 32, 32)

    class DualViewDataset:
        def __init__(self, v1, v2, masks):
            self.v1 = v1
            self.v2 = v2
            self.masks = masks
        def __len__(self):
            return len(self.v1)
        def __getitem__(self, idx):
            return {"image_v1": self.v1[idx], "image_v2": self.v2[idx], "mask": self.masks[idx]}

    return DataLoader(DualViewDataset(v1, v2, masks), batch_size=batch_size, drop_last=True)


def _make_single_view_loader(batch_size=2, num_samples=4):
    images = torch.randn(num_samples, 3, 32, 32)
    masks = torch.rand(num_samples, 1, 32, 32)

    class SingleViewDataset:
        def __init__(self, images, masks):
            self.images = images
            self.masks = masks
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            return {"image": self.images[idx], "mask": self.masks[idx]}

    return DataLoader(SingleViewDataset(images, masks), batch_size=batch_size)


def test_trainer_step_returns_expected_keys():
    model = _make_dummy_model()
    train_loader = _make_dual_view_loader()
    valid_loader = _make_single_view_loader()
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=0.1,
        use_tensorboard=False,
    )
    losses = trainer._step()

    assert "loss" in losses
    assert "seg" in losses
    assert "aux" in losses
    assert "cons" in losses


def test_trainer_tracks_best_val_loss():
    model = _make_dummy_model()
    train_loader = _make_dual_view_loader()
    valid_loader = _make_single_view_loader()
    criterion = SegmentationLoss()
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        cons_criterion=cons_criterion,
        optimizer=optimizer,
        device=device,
        lambda_cons=0.1,
        use_tensorboard=False,
    )

    assert trainer.best_val_loss == float("inf")

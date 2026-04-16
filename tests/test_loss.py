import torch
import torch.nn as nn

from src.ml.training.loss import (
    SymmetricBinaryKLLoss,
    CustomLoss,
    IoULoss,
    SegmentationLoss,
)


def test_iou_loss_zero_for_perfect_overlap():
    pred = torch.ones(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    assert IoULoss()(pred, target).item() < 1e-5


def test_iou_loss_one_for_disjoint():
    pred = torch.zeros(2, 1, 8, 8)
    target = torch.ones(2, 1, 8, 8)
    loss = IoULoss()(pred, target).item()
    assert abs(loss - 1.0) < 1e-3


def test_segmentation_loss_returns_scalar():
    loss_fn = SegmentationLoss()
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    out = loss_fn(pred, target)
    assert out.dim() == 0
    assert out.item() > 0


def test_segmentation_loss_resizes_pred():
    pred = torch.randn(2, 1, 32, 32)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    out = SegmentationLoss()(pred, target)
    assert out.dim() == 0


def test_symmetric_binary_kl_zero_for_identical_logits():
    logits = torch.randn(2, 1, 16, 16)
    loss = SymmetricBinaryKLLoss()(logits, logits)
    assert loss.item() < 1e-6


def test_symmetric_binary_kl_positive_for_different_logits():
    torch.manual_seed(0)
    logits_1 = torch.randn(2, 1, 16, 16)
    logits_2 = torch.randn(2, 1, 16, 16)
    assert SymmetricBinaryKLLoss()(logits_1, logits_2).item() > 0


def test_symmetric_binary_kl_asserts_on_shape_mismatch():
    import pytest

    logits_1 = torch.randn(2, 1, 32, 32)
    logits_2 = torch.randn(2, 1, 16, 16)
    with pytest.raises(AssertionError, match="matching shapes"):
        SymmetricBinaryKLLoss()(logits_1, logits_2)


class _TrainModel(nn.Module):
    """Mimics LoRABiRefNet output: ModelOutput in both modes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        from src.ml.model.lora.wrapper import ModelOutput

        pred = self.conv(x)
        if self.training:
            return ModelOutput(
                preds=[pred, pred], aux=torch.tensor(0.5, device=x.device)
            )
        return ModelOutput(preds=[pred], aux=None)


def test_custom_loss_train_mode_returns_all_terms():
    model = _TrainModel().train()
    batch = {
        "image_1": torch.randn(2, 3, 8, 8),
        "image_2": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
    }
    loss_dict, loss = CustomLoss()(model, batch)
    assert {"loss", "seg", "con", "mae", "aux"} <= set(loss_dict.keys())
    assert torch.isclose(loss, loss_dict["loss"])
    loss.backward()
    assert model.conv.weight.grad is not None


def test_custom_loss_eval_mode_returns_seg_only():
    model = _TrainModel().eval()
    batch = {
        "image_1": torch.randn(2, 3, 8, 8),
        "image_2": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
    }
    loss_dict, loss = CustomLoss()(model, batch)
    assert set(loss_dict.keys()) == {"seg"}
    assert torch.isclose(loss, loss_dict["seg"])

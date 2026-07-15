import torch
import torch.nn as nn

from src.train.loss import (
    AreaConsistencyLoss,
    BoundaryBCELoss,
    SymmetricBinaryKLLoss,
    TrainLoss,
    DiceLoss,
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


def test_region_losses_ignore_padding_and_define_empty_cases():
    target = torch.zeros(2, 1, 4, 4)
    valid = torch.ones_like(target)
    valid[:, :, :, 2:] = 0
    pred = target.clone()
    pred[:, :, :, 2:] = 1
    assert IoULoss()(pred, target, valid).item() == 0.0
    assert DiceLoss()(pred, target, valid).item() == 0.0

    pred[:, :, 0, 0] = 1
    assert IoULoss()(pred, target, valid).item() > 0.9
    assert DiceLoss()(pred, target, valid).item() > 0.9


def test_boundary_bce_ignores_padding():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1
    valid = torch.ones_like(target)
    valid[:, :, :, 6:] = 0
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    reference = BoundaryBCELoss(radius=1)(logits, target, valid)
    logits[:, :, :, 6:] = 100
    assert torch.allclose(BoundaryBCELoss(radius=1)(logits, target, valid), reference)


def test_symmetric_binary_kl_zero_for_identical_logits():
    logits = torch.randn(2, 1, 16, 16)
    loss = SymmetricBinaryKLLoss()(logits, logits)
    assert loss.item() < 1e-6


def test_symmetric_binary_kl_positive_for_different_logits():
    logits_1 = torch.randn(2, 1, 16, 16)
    logits_2 = torch.randn(2, 1, 16, 16)
    assert SymmetricBinaryKLLoss()(logits_1, logits_2).item() > 0


def test_symmetric_binary_kl_asserts_on_shape_mismatch():
    import pytest

    logits_1 = torch.randn(2, 1, 32, 32)
    logits_2 = torch.randn(2, 1, 16, 16)
    with pytest.raises(AssertionError, match="matching shapes"):
        SymmetricBinaryKLLoss()(logits_1, logits_2)


def test_area_consistency_is_samplewise_and_masked():
    logits_1 = torch.zeros(2, 1, 4, 4)
    logits_2 = logits_1.clone()
    valid = torch.ones_like(logits_1)
    valid[:, :, :, 2:] = 0
    logits_2[:, :, :, 2:] = 100
    assert AreaConsistencyLoss()(logits_1, logits_2, valid).item() == 0.0


class _TrainModel(nn.Module):
    """Mimics LoRABiRefNet output: ModelOutput in both modes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        from src.model.lora.model import ModelOutput

        pred = self.conv(x)
        if self.training:
            gdt_pred = pred[:, :, ::2, ::2]
            gdt_label = torch.zeros_like(pred)
            return ModelOutput(
                preds=[pred, pred], gdt=([gdt_pred], [gdt_label])
            )
        return ModelOutput(preds=[pred], gdt=None)


def test_multiscale_boundary_loss_is_not_diluted():
    loss_fn = TrainLoss(lambda_boundary=1.0)
    preds = [torch.zeros(1, 1, size, size) for size in (8, 16, 32, 64)]
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1
    valid = torch.ones_like(mask)
    parts = loss_fn._compute_segments(preds, mask, valid)
    expected = loss_fn.seg.compute(preds[-1], mask, valid)["boundary"]
    assert torch.allclose(parts["boundary"], expected)


def test_gdt_loss_ignores_letterbox_padding():
    loss_fn = TrainLoss()
    valid = torch.ones(1, 1, 8, 8)
    valid[:, :, :, 4:] = 0
    label = torch.zeros(1, 1, 8, 8)
    reference_pred = torch.zeros_like(label)
    changed_pred = reference_pred.clone()
    changed_pred[:, :, :, 4:] = 100
    reference = loss_fn._compute_gdt(([reference_pred], [label]), valid)
    changed = loss_fn._compute_gdt(([changed_pred], [label]), valid)
    assert torch.allclose(changed, reference)


def test_custom_loss_train_mode_returns_all_terms():
    model = _TrainModel().train()
    batch = {
        "image_1": torch.randn(2, 3, 8, 8),
        "image_2": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        "valid_mask": torch.ones(2, 1, 8, 8),
    }
    loss_dict, loss = TrainLoss()(model, batch)
    assert {
        "loss",
        "seg",
        "bce_raw",
        "region_raw",
        "boundary_raw",
        "con_raw",
        "con",
        "area_raw",
        "area",
        "aux_raw",
        "aux",
    } <= set(loss_dict)
    assert torch.isclose(loss, loss_dict["loss"])
    loss.backward()
    assert model.conv.weight.grad is not None


def test_custom_loss_eval_mode_returns_seg_only():
    model = _TrainModel().eval()
    batch = {
        "image_1": torch.randn(2, 3, 8, 8),
        "image_2": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        "valid_mask": torch.ones(2, 1, 8, 8),
    }
    loss_dict, loss = TrainLoss()(model, batch)
    assert {
        "seg",
        "bce_raw",
        "region_raw",
        "boundary_raw",
        "bce",
        "region",
        "boundary",
    } == set(loss_dict)
    assert torch.isclose(loss, loss_dict["seg"])

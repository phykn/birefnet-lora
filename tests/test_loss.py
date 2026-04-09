import torch

from src.finetune.loss import ConsistencyLoss, SegmentationLoss


def test_consistency_loss_zero_for_identical_logits() -> None:
    loss_fn = ConsistencyLoss()
    logits = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits, logits)
    assert loss.item() == 0.0


def test_consistency_loss_positive_for_different_logits() -> None:
    loss_fn = ConsistencyLoss()
    logits1 = torch.randn(2, 1, 32, 32)
    logits2 = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits1, logits2)
    assert loss.item() > 0.0


def test_consistency_loss_interpolates_mismatched_sizes() -> None:
    loss_fn = ConsistencyLoss()
    logits1 = torch.randn(2, 1, 64, 64)
    logits2 = torch.randn(2, 1, 32, 32)
    loss = loss_fn(logits1, logits2)
    assert loss.item() > 0.0


def test_segmentation_loss_returns_scalar() -> None:
    loss_fn = SegmentationLoss()
    pred = torch.randn(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0

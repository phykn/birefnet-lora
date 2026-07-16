import torch
import torch.nn as nn

from src.train.loss import (
    BoundaryBCELoss,
    DiceLoss,
    GCELoss,
    IoULoss,
    SegmentationLoss,
    TrainLoss,
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


def test_gce_is_bounded_for_confident_wrong_label():
    loss = GCELoss(q=0.7)(torch.tensor([[[[-100.0]]]]), torch.ones(1, 1, 1, 1))
    assert loss.item() <= 1.0 / 0.7 + 1e-6


def test_gce_weight_reduces_gt_loss_without_renormalizing():
    logits = torch.zeros(1, 1, 1, 1)
    target = torch.ones_like(logits)
    loss_fn = GCELoss(q=0.7)
    full = loss_fn(logits, target)
    reduced = loss_fn(logits, target, weight=torch.full_like(target, 0.25))
    assert torch.allclose(reduced, full * 0.25)


def test_dice_applies_continuous_weight_once():
    pred = torch.tensor([[[[1.0, 0.0]]]])
    target = torch.ones_like(pred)
    weight = torch.tensor([[[[0.5, 1.0]]]])
    assert torch.allclose(DiceLoss()(pred, target, weight), torch.tensor(0.5))


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


def test_boundary_bce_ignores_only_marked_crop_edges():
    target = torch.ones(1, 1, 8, 8)
    valid = torch.ones_like(target)
    cut = torch.zeros_like(target)
    cut[:, :, 0] = 1
    loss_fn = BoundaryBCELoss(radius=1)

    logits = torch.full_like(target, 20.0)
    reference = loss_fn(logits, target, valid, cut=cut)
    changed = logits.clone()
    changed[:, :, :2] = -20.0
    assert torch.allclose(loss_fn(changed, target, valid, cut=cut), reference)

    changed = logits.clone()
    changed[:, :, -1] = -20.0
    assert loss_fn(changed, target, valid, cut=cut) > reference


class _TrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        from src.adapt.wrap import Output

        pred = self.conv(x)
        if self.training:
            gdt_pred = pred[:, :, ::2, ::2]
            gdt_label = torch.zeros_like(pred)
            return Output(
                logits=[pred, pred], gdt=([gdt_pred], [gdt_label])
            )
        return Output(logits=[pred], gdt=None)


def test_multiscale_boundary_loss_is_not_diluted():
    loss_fn = TrainLoss(lambda_boundary=1.0)
    preds = [torch.zeros(1, 1, size, size) for size in (8, 16, 32, 64)]
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1
    valid = torch.ones_like(mask)
    parts = loss_fn._segment(preds, mask, valid)
    expected = loss_fn.seg.compute(preds[-1], mask, valid)["boundary"]
    assert torch.allclose(parts["boundary"], expected)


def test_gdt_loss_ignores_laplacian_border_around_letterbox():
    loss_fn = TrainLoss()
    valid = torch.ones(1, 1, 8, 8)
    valid[:, :, :, 4:] = 0
    label = torch.zeros(1, 1, 8, 8)
    reference_pred = torch.zeros_like(label)
    changed_pred = reference_pred.clone()
    changed_pred[:, :, :, 2:] = 100
    reference = loss_fn._guide(([reference_pred], [label]), valid)
    changed = loss_fn._guide(([changed_pred], [label]), valid)
    assert torch.allclose(changed, reference)

    changed_pred[:, :, :, 1] = 100
    changed = loss_fn._guide(([changed_pred], [label]), valid)
    assert changed > reference


def test_gdt_target_does_not_receive_gradient():
    loss_fn = TrainLoss()
    pred = torch.zeros(1, 1, 8, 8, requires_grad=True)
    label = torch.zeros(1, 1, 8, 8, requires_grad=True)
    loss_fn._guide(([pred], [label]), torch.ones_like(pred)).backward()
    assert pred.grad is not None
    assert label.grad is None


def test_teacher_only_reduces_conflicting_confident_gt():
    loss_fn = TrainLoss(teacher_confidence=0.95, min_gt_weight=0.25)
    target = torch.zeros(1, 1, 1, 2)
    teacher = torch.tensor([[[[20.0, -20.0]]]])
    weight, confidence, _ = loss_fn._weigh(teacher, target, scale=1.0)
    assert torch.allclose(weight, torch.tensor([[[[0.25, 1.0]]]]))
    assert torch.all(confidence > 0.99)


def test_custom_loss_train_mode_returns_all_terms():
    model = _TrainModel().train()
    batch = {
        "weak": torch.randn(2, 3, 8, 8),
        "strong": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        "valid": torch.ones(2, 1, 8, 8),
    }
    loss_dict, loss = TrainLoss()(model, batch)
    assert {
        "loss",
        "seg",
        "cls_raw",
        "region_raw",
        "boundary_raw",
        "gt_weight",
        "teacher_raw",
        "teacher",
        "aux_raw",
        "aux",
    } <= set(loss_dict)
    assert torch.isclose(loss, loss_dict["loss"])
    loss.backward()
    assert model.conv.weight.grad is not None


def test_custom_loss_eval_mode_returns_seg_only():
    model = _TrainModel().eval()
    batch = {
        "weak": torch.randn(2, 3, 8, 8),
        "strong": torch.randn(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        "valid": torch.ones(2, 1, 8, 8),
    }
    loss_dict, loss = TrainLoss()(model, batch)
    assert {
        "seg",
        "cls_raw",
        "region_raw",
        "boundary_raw",
        "cls",
        "region",
        "boundary",
    } == set(loss_dict)
    assert torch.isclose(loss, loss_dict["seg"])

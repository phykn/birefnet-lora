import torch

from src.ml.training.scheduler import CosineAnnealingWarmupRestarts


def _make(max_lr=1.0, min_lr=0.0, warmup=5, total=20):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.0)
    sched = CosineAnnealingWarmupRestarts(
        optimizer=opt,
        first_cycle_steps=total,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup,
    )
    return opt, sched


def _lr(opt):
    return opt.param_groups[0]["lr"]


def test_starts_at_min_lr():
    opt, _ = _make()
    assert _lr(opt) == 0.0


def test_warmup_reaches_max_lr_after_warmup_steps():
    opt, sched = _make(max_lr=1.0, warmup=5, total=20)
    for _ in range(5):
        sched.step()
    assert abs(_lr(opt) - 1.0) < 1e-6


def test_warmup_is_monotonic_increasing():
    opt, sched = _make(max_lr=1.0, warmup=5, total=20)
    prev = _lr(opt)
    for _ in range(5):
        sched.step()
        cur = _lr(opt)
        assert cur >= prev
        prev = cur


def test_cosine_decays_to_min_lr_at_end_of_cycle():
    opt, sched = _make(max_lr=1.0, min_lr=0.01, warmup=5, total=20)
    for _ in range(20):
        sched.step()
    assert abs(_lr(opt) - 0.01) < 1e-6

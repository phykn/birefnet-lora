import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing schedule with linear warmup and optional warm restarts.

    Args:
        optimizer: Wrapped optimizer.
        first_cycle_steps: Step size of the first cycle.
        cycle_mult: Cycle length multiplier applied at each restart.
        max_lr: Max learning rate of the first cycle.
        min_lr: Minimum learning rate.
        warmup_steps: Linear warmup step size (must be < first_cycle_steps).
        gamma: Per-cycle decay factor applied to max_lr.
        last_epoch: Index of the last epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        progress = (self.step_in_cycle - self.warmup_steps) / (
            self.cur_cycle_steps - self.warmup_steps
        )
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is not None:
            raise TypeError(
                "CosineAnnealingWarmupRestarts does not support epoch-resume; "
                "call step() without arguments."
            )
        self.last_epoch += 1
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle -= self.cur_cycle_steps
            self.cur_cycle_steps = (
                int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                + self.warmup_steps
            )

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

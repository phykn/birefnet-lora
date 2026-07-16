import torch
import torch.nn as nn

from src.adapt.wrap import Output
from src.train.teacher import Teacher


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> Output:
        return Output(logits=[x * self.weight + self.frozen])


def test_teacher_tracks_only_trainable_params_and_restores_mode():
    model = _Model().train()
    teacher = Teacher(model, decay=0.5, start=2, ramp=2)
    assert set(teacher.params) == {"weight"}

    with torch.no_grad():
        model.weight.fill_(3.0)
    pred = teacher.predict(model, torch.ones(1, 1, 1, 1))
    assert torch.allclose(pred, torch.tensor([[[[3.0]]]]))
    assert model.training

    teacher.update(model)
    assert torch.allclose(teacher.params["weight"], torch.tensor(2.0))


def test_teacher_scale_warms_up_after_start():
    teacher = Teacher(_Model(), start=2, ramp=2)
    assert teacher.scale(2) == 0.0
    assert teacher.scale(3) == 0.5
    assert teacher.scale(4) == 1.0

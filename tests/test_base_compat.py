from pathlib import Path

import pytest
import torch

from src.model.birefnet import BiRefNet

ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = ROOT / "weight" / "BiRefNet-general-epoch_244.pth"


@pytest.mark.skipif(not BASE_CHECKPOINT.exists(), reason="base checkpoint is absent")
def test_pinned_base_checkpoint_strict_loads():
    model = BiRefNet([1536, 768, 384, 192], grad_checkpoint=False)
    state = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_raw_base_forward_keeps_upstream_train_and_eval_container_contract():
    with torch.device("meta"):
        model = BiRefNet([1536, 768, 384, 192], grad_checkpoint=False)
        pred = torch.empty(1, 1, 4, 4, device="meta")
        gdt = [[pred], [pred]]
        scaled = [gdt, [pred]]

    model.predict = lambda x: (scaled, None)
    x = torch.empty(1, 3, 4, 4, device="meta")

    model.eval()
    assert model(x) is scaled

    model.train()
    output = model(x)
    assert output[0] is scaled
    assert output[1] == [None]

import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def api_client():
    from run_api import build_app

    def _build(
        model,
        device: torch.device,
        threshold: float | None = None,
    ) -> TestClient:
        app = build_app(
            model=model,
            device=device,
            settings={
                "size": 32,
                "mode": "rgb",
                "overlap_ratio": 1 / 3,
                "tile_batch": 2,
                "context_weight": 0.0,
                "default_threshold": 0.5,
            },
            threshold=threshold,
        )
        return TestClient(app)

    return _build

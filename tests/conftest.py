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
    from src.serve.app import build_app

    def _build(
        model,
        device: torch.device,
        threshold: float | None = None,
        preprocess=None,
    ) -> TestClient:
        app = build_app(
            model=model,
            device=device,
            threshold=threshold,
            preprocess=preprocess,
        )
        return TestClient(app)

    return _build

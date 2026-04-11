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

    def _build(model, device: torch.device) -> TestClient:
        app = build_app(model=model, device=device, max_concurrency=2)
        return TestClient(app)

    return _build

import asyncio
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
    from src.api.app import app

    def _build(model, device: torch.device) -> TestClient:
        app.state.model = model
        app.state.device = device
        app.state.predict_sem = asyncio.Semaphore(2)
        return TestClient(app)

    return _build

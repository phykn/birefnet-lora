import torch
from fastapi.testclient import TestClient

from src.api.app import create_app


def test_health_returns_ok_and_device():
    app = create_app(model=None, device=torch.device("cpu"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "device": "cpu"}

import torch


def test_health_returns_ok_and_device(api_client):
    client = api_client(model=None, device=torch.device("cpu"))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "device": "cpu"}

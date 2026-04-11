import numpy as np
import torch
from fastapi.testclient import TestClient
from imstr import decode, encode

from src.api.app import create_app


def _make_client(monkeypatch, fake_predict):
    monkeypatch.setattr(
        "src.api.routes.predict.predict", fake_predict, raising=True
    )
    app = create_app(model=object(), device=torch.device("cpu"))
    return TestClient(app)


def test_predict_binary_mask_default_threshold(monkeypatch):
    captured = {}

    def fake_predict(model, image, threshold=0.5):
        captured["threshold"] = threshold
        captured["shape"] = image.shape
        return np.full(image.shape[:2], 255, dtype=np.uint8)

    client = _make_client(monkeypatch, fake_predict)

    img = np.random.randint(0, 255, (16, 24, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    mask = decode(response.json()["mask"])
    assert mask.dtype == np.uint8
    assert mask.shape[:2] == (16, 24)
    assert (mask == 255).all()
    assert captured["threshold"] == 0.5
    assert captured["shape"] == (16, 24, 3)


def test_predict_releases_cuda_cache_on_cuda_device(monkeypatch):
    def fake_predict(model, image, threshold=0.5):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        "src.api.routes.predict.predict", fake_predict, raising=True
    )

    calls = {"empty_cache": 0}

    def fake_empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    # torch.device("cuda") is just a marker — we never touch real CUDA
    app = create_app(model=object(), device=torch.device("cuda"))
    client = TestClient(app)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    assert calls["empty_cache"] == 1


def test_predict_skips_empty_cache_on_cpu_device(monkeypatch):
    def fake_predict(model, image, threshold=0.5):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(
        "src.api.routes.predict.predict", fake_predict, raising=True
    )

    calls = {"empty_cache": 0}

    def fake_empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    app = create_app(model=object(), device=torch.device("cpu"))
    client = TestClient(app)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    assert calls["empty_cache"] == 0


def test_predict_probability_map_when_threshold_negative(monkeypatch):
    def fake_predict(model, image, threshold=0.5):
        assert threshold is None
        return np.full(image.shape[:2], 0.5, dtype=np.float32)

    client = _make_client(monkeypatch, fake_predict)

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    response = client.post(
        "/predict", json={"image": encode(img), "threshold": -1.0}
    )

    assert response.status_code == 200
    mask = decode(response.json()["mask"])
    assert mask.dtype == np.uint8
    # 0.5 * 255 -> 127 (astype truncation)
    assert mask.shape[:2] == (8, 8)
    assert (mask == 127).all()


def test_predict_rejects_invalid_image(monkeypatch):
    def fake_predict(model, image, threshold=0.5):
        raise AssertionError("predict should not be called on invalid input")

    client = _make_client(monkeypatch, fake_predict)

    response = client.post("/predict", json={"image": "not-a-real-imstr"})

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid image"}


def test_predict_rejects_threshold_out_of_range(monkeypatch):
    def fake_predict(model, image, threshold=0.5):
        raise AssertionError("should not reach predict")

    client = _make_client(monkeypatch, fake_predict)

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post(
        "/predict", json={"image": encode(img), "threshold": 2.0}
    )

    assert response.status_code == 422

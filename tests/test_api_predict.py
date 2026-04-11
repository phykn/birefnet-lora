import numpy as np
import torch
from imstr import decode, encode

PREDICT_TARGET = "src.api.routes.run_predict"


def _patch_predict(monkeypatch, fake):
    monkeypatch.setattr(PREDICT_TARGET, fake, raising=True)


def test_predict_binary_mask_default_threshold(monkeypatch, api_client):
    captured = {}

    def fake_predict(model, image, threshold=0.5):
        captured["threshold"] = threshold
        captured["shape"] = image.shape
        return np.full(image.shape[:2], 255, dtype=np.uint8)

    _patch_predict(monkeypatch, fake_predict)
    client = api_client(model=object(), device=torch.device("cpu"))

    img = np.random.randint(0, 255, (16, 24, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    mask = decode(response.json()["mask"])
    assert mask.dtype == np.uint8
    assert mask.shape[:2] == (16, 24)
    assert (mask == 255).all()
    assert captured["threshold"] == 0.5
    assert captured["shape"] == (16, 24, 3)


def test_predict_releases_cuda_cache_on_cuda_device(monkeypatch, api_client):
    def fake_predict(model, image, threshold=0.5):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    _patch_predict(monkeypatch, fake_predict)

    calls = {"empty_cache": 0}

    def fake_empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    client = api_client(model=object(), device=torch.device("cuda"))

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    assert calls["empty_cache"] == 1


def test_predict_skips_empty_cache_on_cpu_device(monkeypatch, api_client):
    def fake_predict(model, image, threshold=0.5):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    _patch_predict(monkeypatch, fake_predict)

    calls = {"empty_cache": 0}

    def fake_empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    client = api_client(model=object(), device=torch.device("cpu"))

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post("/predict", json={"image": encode(img)})

    assert response.status_code == 200
    assert calls["empty_cache"] == 0


def test_predict_probability_map_when_threshold_negative(monkeypatch, api_client):
    def fake_predict(model, image, threshold=0.5):
        assert threshold is None
        return np.full(image.shape[:2], 0.5, dtype=np.float32)

    _patch_predict(monkeypatch, fake_predict)
    client = api_client(model=object(), device=torch.device("cpu"))

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    response = client.post(
        "/predict", json={"image": encode(img), "threshold": -1.0}
    )

    assert response.status_code == 200
    mask = decode(response.json()["mask"])
    assert mask.dtype == np.uint8
    assert mask.shape[:2] == (8, 8)
    assert (mask == 127).all()


def test_predict_rejects_invalid_image(monkeypatch, api_client):
    def fake_predict(model, image, threshold=0.5):
        raise AssertionError("predict should not be called on invalid input")

    _patch_predict(monkeypatch, fake_predict)
    client = api_client(model=object(), device=torch.device("cpu"))

    response = client.post("/predict", json={"image": "not-a-real-imstr"})

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid image"}


def test_predict_rejects_threshold_out_of_range(monkeypatch, api_client):
    def fake_predict(model, image, threshold=0.5):
        raise AssertionError("should not reach predict")

    _patch_predict(monkeypatch, fake_predict)
    client = api_client(model=object(), device=torch.device("cpu"))

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    response = client.post(
        "/predict", json={"image": encode(img), "threshold": 2.0}
    )

    assert response.status_code == 422

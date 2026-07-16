import base64

import cv2
import numpy as np
import pytest
import torch

PREDICT_TARGET = "src.serve.route.predict_mask"


def _encode(image: np.ndarray) -> str:
    source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    return base64.b64encode(encoded).decode("ascii")


def _decode(value: str) -> np.ndarray:
    encoded = base64.b64decode(value)
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)


def test_defaults_to_binary_and_uses_saved_threshold(monkeypatch, api_client):
    captured = {}

    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        return np.full(image.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(object(), torch.device("cpu"), threshold=0.62)
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    response = client.post(
        "/predict",
        json={"id": "sample", "base64_str": _encode(image)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "sample"
    assert payload["output_mode"] == "binary"
    assert payload["threshold_applied"] == 0.62
    assert payload["dtype"] == "uint8"
    assert payload["value_range"] == [0, 255]
    assert _decode(payload["base64_str"]).shape == (16, 24)
    assert captured["output_mode"] == "binary"
    assert captured["threshold"] == 0.62
    assert captured["tiles"] == (1,)
    assert captured["overlap"] == pytest.approx(1 / 3)


def test_request_threshold_overrides_saved_value(monkeypatch, api_client):
    captured = {}

    def fake_predict(model, image, **kwargs):
        captured.update(kwargs)
        return np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(object(), torch.device("cpu"), threshold=0.62)
    response = client.post(
        "/predict",
        json={
            "base64_str": _encode(np.zeros((4, 4, 3), np.uint8)),
            "threshold": 0.4,
            "tiles": [1, 4],
            "overlap": 0.4,
        },
    )
    assert response.status_code == 200
    assert captured["threshold"] == 0.4
    assert captured["tiles"] == (1, 4)
    assert captured["overlap"] == 0.4
    assert response.json()["threshold_applied"] == 0.4


def test_probability_mode_is_explicit(monkeypatch, api_client):
    def fake_predict(model, image, **kwargs):
        assert kwargs["output_mode"] == "probability"
        return np.full(image.shape[:2], 127, dtype=np.uint8)

    monkeypatch.setattr(PREDICT_TARGET, fake_predict)
    client = api_client(object(), torch.device("cpu"))
    response = client.post(
        "/predict",
        json={
            "base64_str": _encode(np.zeros((8, 8, 3), np.uint8)),
            "output_mode": "probability",
        },
    )
    assert response.status_code == 200
    assert response.json()["threshold_applied"] is None
    assert np.all(_decode(response.json()["base64_str"]) == 127)


def test_rejects_invalid_image(monkeypatch, api_client):
    monkeypatch.setattr(
        PREDICT_TARGET,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    client = api_client(object(), torch.device("cpu"))
    response = client.post("/predict", json={"base64_str": "invalid"})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid image"}


def test_rejects_invalid_output_contract(api_client):
    client = api_client(object(), torch.device("cpu"))
    encoded = _encode(np.zeros((4, 4, 3), np.uint8))
    assert (
        client.post(
            "/predict", json={"base64_str": encoded, "threshold": 2.0}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"base64_str": encoded, "output_mode": "soft"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"base64_str": encoded, "tiles": [0]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"base64_str": encoded, "overlap": 0.2}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    ("name", "limit"),
    [("MAX_BYTES", 1), ("MAX_PIXELS", 15)],
)
def test_rejects_large_image(monkeypatch, api_client, name, limit):
    monkeypatch.setattr(
        f"src.serve.codec.{name}",
        limit,
    )
    monkeypatch.setattr(
        PREDICT_TARGET,
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    client = api_client(object(), torch.device("cpu"))
    response = client.post(
        "/predict",
        json={"base64_str": _encode(np.zeros((4, 4, 3), np.uint8))},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "image too large"}

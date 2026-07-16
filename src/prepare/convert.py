from typing import Literal

import cv2
import numpy as np

InputMode = Literal["rgb", "gray_repeat", "gray_features"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return (x - MEAN) / STD


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got shape {image.shape}")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[..., :3]
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)


def _zscore(gray: np.ndarray, std_range: float = 3.0) -> np.ndarray:
    x = gray.astype(np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    low, high = mean - std_range * std, mean + std_range * std
    x = np.clip(x, low, high)
    x = (x - low) / max(high - low, 1e-6)
    return np.rint(x * 255.0).astype(np.uint8)


def _enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(_zscore(gray))
    return clahe.apply(_zscore(x))


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def convert(image: np.ndarray, mode: InputMode = "rgb") -> np.ndarray:
    rgb = _to_rgb(image)
    if mode == "rgb":
        return rgb
    gray = _to_gray(rgb)
    if mode == "gray_repeat":
        return np.repeat(gray[..., None], 3, axis=2)
    if mode == "gray_features":
        return np.stack([gray, _enhance(gray), _sharpen(gray)], axis=-1)
    raise ValueError(f"Unsupported input mode: {mode!r}")

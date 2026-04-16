import cv2
import numpy as np

from sdimg.image import to_gray, zscore_norm, clahe_norm

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return (x - MEAN) / STD


def _norm(gray: np.ndarray) -> np.ndarray:
    x = zscore_norm(gray, std_range=3.0)
    x = clahe_norm(x, clipLimit=2.0, tileGridSize=(8, 8))
    x = zscore_norm(x, std_range=3.0)
    x = clahe_norm(x, clipLimit=2.0, tileGridSize=(8, 8))
    return x


def _sharpen(gray: np.ndarray) -> np.ndarray:
    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    except Exception:
        try:
            blur = cv2.medianBlur(gray, 5)
        except Exception:
            return gray.copy()

    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return np.stack([gray, _norm(gray), _sharpen(gray)], axis=-1)

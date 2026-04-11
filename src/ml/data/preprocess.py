import cv2
import numpy as np

from sdimg.image import to_gray, zscore_norm, clahe_norm


def _norm(gray: np.ndarray) -> np.ndarray:
    x = zscore_norm(gray, std_range=3.0)
    x = clahe_norm(x, clip_limit=2.0, tile_grid_size=(8, 8))
    x = zscore_norm(x, std_range=3.0)
    x = clahe_norm(x, clip_limit=2.0, tile_grid_size=(8, 8))
    return x


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return np.stack([gray, _norm(gray), _sharpen(gray)], axis=-1)

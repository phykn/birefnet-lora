import cv2
import numpy as np

from sdimg.image import to_gray

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    clahe = _CLAHE.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return np.stack([gray, clahe, sharp], axis=-1)

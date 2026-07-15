from dataclasses import dataclass

import cv2
import numpy as np

from .input import InputMode, convert, normalize


@dataclass(frozen=True)
class LetterboxGeometry:
    original_h: int
    original_w: int
    resized_h: int
    resized_w: int
    pad_top: int
    pad_left: int
    canvas_size: int


def plan(height: int, width: int, size: int = 1024) -> LetterboxGeometry:
    if height <= 0 or width <= 0 or size <= 0:
        raise ValueError(
            f"height, width and size must be positive, got {(height, width, size)}"
        )
    scale = min(size / height, size / width)
    resized_h = min(size, max(1, int(round(height * scale))))
    resized_w = min(size, max(1, int(round(width * scale))))
    return LetterboxGeometry(
        original_h=height,
        original_w=width,
        resized_h=resized_h,
        resized_w=resized_w,
        pad_top=(size - resized_h) // 2,
        pad_left=(size - resized_w) // 2,
        canvas_size=size,
    )


def _resize(
    image: np.ndarray,
    geometry: LetterboxGeometry,
    interpolation: int,
) -> np.ndarray:
    return cv2.resize(
        image,
        (geometry.resized_w, geometry.resized_h),
        interpolation=interpolation,
    )


def prepare_image(
    image: np.ndarray,
    size: int = 1024,
    mode: InputMode = "rgb",
    geometry: LetterboxGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray, LetterboxGeometry]:
    geometry = geometry or plan(*image.shape[:2], size=size)
    processed = convert(image, mode=mode)
    downsampling = (
        geometry.resized_h < image.shape[0] or geometry.resized_w < image.shape[1]
    )
    interpolation = cv2.INTER_AREA if downsampling else cv2.INTER_CUBIC
    resized = normalize(_resize(processed, geometry, interpolation))

    canvas = np.zeros((size, size, 3), dtype=np.float32)
    valid = np.zeros((1, size, size), dtype=np.float32)
    y0, x0 = geometry.pad_top, geometry.pad_left
    y1, x1 = y0 + geometry.resized_h, x0 + geometry.resized_w
    canvas[y0:y1, x0:x1] = resized
    valid[:, y0:y1, x0:x1] = 1.0
    return np.transpose(canvas, (2, 0, 1)), valid, geometry


def prepare_mask(mask: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    resized = _resize(mask.astype(np.float32), geometry, cv2.INTER_NEAREST_EXACT)
    canvas = np.zeros((geometry.canvas_size, geometry.canvas_size), dtype=np.float32)
    y0, x0 = geometry.pad_top, geometry.pad_left
    y1, x1 = y0 + geometry.resized_h, x0 + geometry.resized_w
    canvas[y0:y1, x0:x1] = resized
    return canvas[None]


def restore_logit(logit: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    y0, x0 = geometry.pad_top, geometry.pad_left
    y1, x1 = y0 + geometry.resized_h, x0 + geometry.resized_w
    cropped = logit[y0:y1, x0:x1]
    if cropped.shape == (geometry.original_h, geometry.original_w):
        return cropped.astype(np.float32, copy=False)
    return cv2.resize(
        cropped.astype(np.float32),
        (geometry.original_w, geometry.original_h),
        interpolation=cv2.INTER_LINEAR,
    )

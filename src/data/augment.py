import random

import cv2
import numpy as np


def flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_k = int(np.random.randint(4))
    do_hflip = bool(np.random.randint(2))
    return _flip(image, rot_k, do_hflip), _flip(mask, rot_k, do_hflip)


def _flip(x: np.ndarray, rot_k: int, do_hflip: bool) -> np.ndarray:
    x = np.rot90(x, k=rot_k)
    if do_hflip:
        x = np.fliplr(x)
    return np.ascontiguousarray(x)


def crop(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    global_prob: float,
    boundary_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if random.random() < global_prob or (height <= size and width <= size):
        return image, mask

    crop_h = min(size, height)
    crop_w = min(size, width)
    if random.random() < boundary_prob:
        binary = (mask > 127).astype(np.uint8)
        boundary = cv2.morphologyEx(
            binary,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        ys, xs = np.nonzero(boundary)
    else:
        ys = xs = np.empty(0, dtype=np.int64)

    if len(ys):
        pick = random.randrange(len(ys))
        center_y, center_x = int(ys[pick]), int(xs[pick])
    else:
        center_y = random.randrange(height)
        center_x = random.randrange(width)

    top = min(max(center_y - crop_h // 2, 0), height - crop_h)
    left = min(max(center_x - crop_w // 2, 0), width - crop_w)
    bottom, right = top + crop_h, left + crop_w
    return (
        image[top:bottom, left:right].copy(),
        mask[top:bottom, left:right].copy(),
    )


def jitter(image: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    brightness_max, contrast_max = value_range
    brightness = random.uniform(-brightness_max, brightness_max)
    contrast = random.uniform(-contrast_max, contrast_max)
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) * (1.0 + contrast) + 0.5 + brightness
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

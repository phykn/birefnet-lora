import random
import numpy as np

from sdimg.spatial import crop


def random_box(hw: tuple[int, int], scale: float) -> list[int]:
    assert 0 < scale <= 1, "Scale must be in the range (0, 1]"
    img_h, img_w = hw
    size_w = int(img_w * scale)
    size_h = int(img_h * scale)

    wmin = random.randint(0, img_w - size_w)
    hmin = random.randint(0, img_h - size_h)
    wmax = wmin + size_w
    hmax = hmin + size_h
    return [wmin, hmin, wmax, hmax]


def random_crop(
    image: np.ndarray, mask: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    assert image.shape[:2] == mask.shape[:2]
    box = random_box(hw=image.shape[:2], scale=scale)
    return crop(image, box), crop(mask, box)

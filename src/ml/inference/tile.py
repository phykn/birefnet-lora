import math

import numpy as np


def _patch_axis(length: int, n: int, overlap: float) -> tuple[list[int], int]:
    if n == 1:
        return [0], length
    patch_size = math.ceil(length / (1 + (n - 1) * (1 - overlap)))
    patch_size = min(patch_size, length)
    span = length - patch_size
    starts = np.rint(np.linspace(0, span, num=n)).astype(int).tolist()
    return starts, patch_size


def split(
    src: np.ndarray,
    n_w: int,
    n_h: int,
    overlap: float = 0.2,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    h, w = src.shape[:2]
    starts_h, ph = _patch_axis(h, n_h, overlap)
    starts_w, pw = _patch_axis(w, n_w, overlap)

    patches: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    for hmin in starts_h:
        hmax = hmin + ph
        for wmin in starts_w:
            wmax = wmin + pw
            patches.append(src[hmin:hmax, wmin:wmax].copy())
            boxes.append((hmin, wmin, hmax, wmax))
    return patches, boxes


def _cosine_window(h: int, w: int) -> np.ndarray:
    wh = np.ones(h, dtype=np.float32)
    ww = np.ones(w, dtype=np.float32)
    if h > 1:
        wh = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, h, dtype=np.float32))
        np.maximum(wh, 1e-3, out=wh)
    if w > 1:
        ww = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, w, dtype=np.float32))
        np.maximum(ww, 1e-3, out=ww)
    return wh[:, None] * ww[None, :]


def merge_logits(
    patches: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    merged = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    cache: dict[tuple[int, int], np.ndarray] = {}

    for patch, (hmin, wmin, hmax, wmax) in zip(patches, boxes):
        ph, pw = hmax - hmin, wmax - wmin
        key = (ph, pw)
        if key not in cache:
            cache[key] = _cosine_window(ph, pw)
        win = cache[key]
        merged[hmin:hmax, wmin:wmax] += patch * win
        weights[hmin:hmax, wmin:wmax] += win

    np.maximum(weights, 1e-6, out=weights)
    merged /= weights
    return merged

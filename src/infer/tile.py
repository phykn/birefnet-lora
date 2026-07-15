import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileBox:
    top: int
    left: int
    bottom: int
    right: int
    overlap_top: int = 0
    overlap_left: int = 0
    overlap_bottom: int = 0
    overlap_right: int = 0

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def _plan_axis(
    length: int, count: int, overlap_ratio: float
) -> list[tuple[int, int, int, int]]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if count == 1:
        return [(0, length, 0, 0)]

    extent = math.ceil(length / (count - (count - 1) * overlap_ratio))
    while True:
        starts = np.rint(np.linspace(0, length - extent, count)).astype(int)
        min_overlap = math.ceil(extent * overlap_ratio)
        actual = extent - int(np.diff(starts).max())
        if actual >= min_overlap:
            break
        extent += 1

    planned = []
    for index, start in enumerate(starts):
        start = int(start)
        end = start + extent
        before = 0 if index == 0 else int(starts[index - 1]) + extent - start
        after = 0 if index == len(starts) - 1 else end - int(starts[index + 1])
        planned.append((start, end, before, after))
    return planned


def plan_tiles(
    height: int,
    width: int,
    size: int = 1024,
    overlap_ratio: float = 1 / 3,
) -> list[TileBox]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")

    longest = max(height, width)
    if longest <= size:
        grid = 1
    elif longest <= size * (2 - overlap_ratio):
        grid = 2
    else:
        grid = 3

    ys = _plan_axis(height, grid, overlap_ratio)
    xs = _plan_axis(width, grid, overlap_ratio)
    return [
        TileBox(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            overlap_top=overlap_top,
            overlap_left=overlap_left,
            overlap_bottom=overlap_bottom,
            overlap_right=overlap_right,
        )
        for top, bottom, overlap_top, overlap_bottom in ys
        for left, right, overlap_left, overlap_right in xs
    ]


def _make_ramp(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    angles = np.linspace(0.0, np.pi, length + 2, dtype=np.float32)[1:-1]
    return 0.5 - 0.5 * np.cos(angles)


def make_window(box: TileBox) -> np.ndarray:
    wy = np.ones(box.height, dtype=np.float32)
    wx = np.ones(box.width, dtype=np.float32)
    if box.overlap_top:
        wy[: box.overlap_top] = np.minimum(
            wy[: box.overlap_top], _make_ramp(box.overlap_top)
        )
    if box.overlap_bottom:
        wy[-box.overlap_bottom :] = np.minimum(
            wy[-box.overlap_bottom :],
            _make_ramp(box.overlap_bottom)[::-1],
        )
    if box.overlap_left:
        wx[: box.overlap_left] = np.minimum(
            wx[: box.overlap_left], _make_ramp(box.overlap_left)
        )
    if box.overlap_right:
        wx[-box.overlap_right :] = np.minimum(
            wx[-box.overlap_right :],
            _make_ramp(box.overlap_right)[::-1],
        )
    return wy[:, None] * wx[None, :]

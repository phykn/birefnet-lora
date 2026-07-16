import contextlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from ..prepare.convert import InputMode, convert
from ..prepare.fit import fit_image, restore
from .tile import plan, weigh

OutputMode = Literal["binary", "probability"]


def _autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )
    return torch.amp.autocast("cuda", dtype=dtype)


def _infer(
    model: torch.nn.Module,
    tensors: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with _autocast(device):
        logits = model(batch).logits[-1]
    return logits[:, 0].float().cpu().numpy()


def _merge(
    model: torch.nn.Module,
    image: np.ndarray,
    grid: int,
    size: int,
    overlap: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    height, width = image.shape[:2]
    boxes = plan(height, width, grid=grid, overlap=overlap)
    merged = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start : start + batch_size]
        tensors = []
        fits = []
        for box in chunk:
            crop = image[box.top : box.bottom, box.left : box.right]
            tensor, _, fit = fit_image(crop, size=size, mode="rgb")
            tensors.append(tensor)
            fits.append(fit)

        logits = _infer(model, tensors, device)
        for index, (box, fit) in enumerate(zip(chunk, fits)):
            blend = weigh(box)
            region = np.s_[box.top : box.bottom, box.left : box.right]
            merged[region] += restore(logits[index], fit) * blend
            weight[region] += blend

    if np.any(weight <= 0):
        raise RuntimeError("Tile planner left pixels without merge weight")
    return merged / weight


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int = 1024,
    mode: InputMode = "rgb",
    tiles: Sequence[int] = (1,),
    overlap: float = 1 / 3,
    tile_batch: int = 2,
) -> np.ndarray:
    grids = tuple(tiles)
    if not grids:
        raise ValueError("tiles must not be empty")
    if any(
        not isinstance(grid, int) or isinstance(grid, bool) or grid <= 0
        for grid in grids
    ):
        raise ValueError("tiles must contain positive integers")
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    if not 1 / 3 <= overlap < 1.0:
        raise ValueError("overlap must be in [1/3, 1)")

    model.eval()
    device = next(model.parameters()).device
    processed = convert(image, mode=mode)
    outputs = []
    for grid in grids:
        if grid == 1:
            tensor, _, fit = fit_image(processed, size=size, mode="rgb")
            output = restore(_infer(model, [tensor], device)[0], fit)
        else:
            output = _merge(
                model,
                processed,
                grid,
                size,
                overlap,
                tile_batch,
                device,
            )
        outputs.append(output)

    return np.mean(outputs, axis=0, dtype=np.float32)


def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    output_mode: OutputMode = "binary",
    threshold: float | None = None,
    size: int = 1024,
    mode: InputMode = "rgb",
    tiles: Sequence[int] = (1,),
    overlap: float = 1 / 3,
    tile_batch: int = 2,
) -> np.ndarray:
    if output_mode not in {"binary", "probability"}:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    grids = tuple(tiles)
    if output_mode == "binary" and threshold is None:
        if any(grid != 1 for grid in grids):
            raise ValueError("threshold is required for tiled binary output")
        threshold = 0.5

    logits = predict_logits(
        model,
        image,
        size=size,
        mode=mode,
        tiles=grids,
        overlap=overlap,
        tile_batch=tile_batch,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    if output_mode == "probability":
        return np.rint(probability * 255.0).astype(np.uint8)
    assert threshold is not None
    return (probability >= threshold).astype(np.uint8) * 255

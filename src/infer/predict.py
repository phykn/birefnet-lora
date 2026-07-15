import contextlib
from typing import Literal

import numpy as np
import torch

from ..data.input import InputMode, convert
from ..data.letterbox import prepare_image, restore_logit
from .tile import make_window, plan_tiles

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


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int = 1024,
    mode: InputMode = "rgb",
    overlap_ratio: float = 1 / 3,
    tile_batch: int = 2,
    context_weight: float = 0.0,
) -> np.ndarray:
    if tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("context_weight must be in [0, 1]")

    model.eval()
    device = next(model.parameters()).device
    processed = convert(image, mode=mode)
    height, width = processed.shape[:2]
    boxes = plan_tiles(height, width, size=size, overlap_ratio=overlap_ratio)
    merged = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(boxes), tile_batch):
        chunk = boxes[start : start + tile_batch]
        prepared = []
        geometries = []
        for box in chunk:
            tile = processed[box.top : box.bottom, box.left : box.right]
            tensor, _, geometry = prepare_image(tile, size=size, mode="rgb")
            prepared.append(tensor)
            geometries.append(geometry)

        batch = torch.from_numpy(np.stack(prepared)).to(device)
        with _autocast(device):
            logits = model(batch).preds[-1]

        for index, (box, geometry) in enumerate(zip(chunk, geometries)):
            canvas_logit = logits[index, 0].float().cpu().numpy()
            tile_logit = restore_logit(canvas_logit, geometry)
            window = make_window(box)
            region = np.s_[box.top : box.bottom, box.left : box.right]
            merged[region] += tile_logit * window
            weights[region] += window

    if np.any(weights <= 0):
        raise RuntimeError("Tile planner left pixels without merge weight")
    merged /= weights

    if context_weight > 0.0 and len(boxes) > 1:
        full_tensor, _, geometry = prepare_image(
            processed,
            size=size,
            mode="rgb",
        )
        full_batch = torch.from_numpy(full_tensor[None]).to(device)
        with _autocast(device):
            full_canvas = model(full_batch).preds[-1][0, 0].float().cpu().numpy()
        full_logit = restore_logit(full_canvas, geometry)
        merged = (1.0 - context_weight) * merged + context_weight * full_logit

    return merged


def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    output_mode: OutputMode = "binary",
    threshold: float = 0.5,
    size: int = 1024,
    mode: InputMode = "rgb",
    overlap_ratio: float = 1 / 3,
    tile_batch: int = 2,
    context_weight: float = 0.0,
) -> np.ndarray:
    if output_mode not in {"binary", "probability"}:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    logits = predict_logits(
        model,
        image,
        size=size,
        mode=mode,
        overlap_ratio=overlap_ratio,
        tile_batch=tile_batch,
        context_weight=context_weight,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    if output_mode == "probability":
        return np.rint(probability * 255.0).astype(np.uint8)
    return (probability >= threshold).astype(np.uint8) * 255

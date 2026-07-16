import contextlib
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


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    size: int = 1024,
    mode: InputMode = "rgb",
    tile: bool = False,
    overlap_ratio: float = 1 / 3,
    tile_batch: int = 2,
    context_weight: float = 0.0,
) -> np.ndarray:
    if tile and tile_batch <= 0:
        raise ValueError("tile_batch must be positive")
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("context_weight must be in [0, 1]")
    if not tile and context_weight > 0.0:
        raise ValueError("context_weight requires tile=True")

    model.eval()
    device = next(model.parameters()).device
    processed = convert(image, mode=mode)
    height, width = processed.shape[:2]
    if not tile:
        tensor, _, fit = fit_image(processed, size=size, mode="rgb")
        return restore(_infer(model, [tensor], device)[0], fit)

    tiles = plan(height, width, size=size, overlap_ratio=overlap_ratio)
    merged = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(tiles), tile_batch):
        chunk = tiles[start : start + tile_batch]
        tensors = []
        fits = []
        for box in chunk:
            crop = processed[box.top : box.bottom, box.left : box.right]
            tensor, _, fit = fit_image(crop, size=size, mode="rgb")
            tensors.append(tensor)
            fits.append(fit)

        logits = _infer(model, tensors, device)

        for index, (box, fit) in enumerate(zip(chunk, fits)):
            canvas = logits[index]
            tile_logit = restore(canvas, fit)
            blend = weigh(box)
            region = np.s_[box.top : box.bottom, box.left : box.right]
            merged[region] += tile_logit * blend
            weight[region] += blend

    if np.any(weight <= 0):
        raise RuntimeError("Tile planner left pixels without merge weight")
    merged /= weight

    if context_weight > 0.0 and len(tiles) > 1:
        full_tensor, _, fit = fit_image(
            processed,
            size=size,
            mode="rgb",
        )
        full_canvas = _infer(model, [full_tensor], device)[0]
        full_logit = restore(full_canvas, fit)
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
    tile: bool = False,
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
        tile=tile,
        overlap_ratio=overlap_ratio,
        tile_batch=tile_batch,
        context_weight=context_weight,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    if output_mode == "probability":
        return np.rint(probability * 255.0).astype(np.uint8)
    return (probability >= threshold).astype(np.uint8) * 255

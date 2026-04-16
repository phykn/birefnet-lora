import contextlib

import cv2
import numpy as np
import torch

from sdimg.spatial import resize

from ..preprocess import normalize, preprocess
from .tile import merge_logits, split


def _prepare(image: np.ndarray, size: int) -> np.ndarray:
    x = resize(image, height=size, width=size, interpolation=cv2.INTER_CUBIC)
    x = normalize(x)
    return np.transpose(x, (2, 0, 1))


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    size: int = 1024,
    threshold: float | None = None,
    n: int | tuple[int, int] = 1,
    overlap: float = 0.1,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    h, w = image.shape[:2]
    n_w, n_h = (n, n) if isinstance(n, int) else n

    # 1. split → preprocess → prepare → batch
    patches, boxes = split(image, n_w, n_h, overlap=overlap)
    batch = np.stack([_prepare(preprocess(p), size) for p in patches])
    x = torch.from_numpy(batch).to(device)

    # 2. inference
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float16
        )
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    with autocast_ctx:
        logits = model(x).preds[-1]

    # 3. interpolate each logit back to patch size → merge → sigmoid
    logit_patches = []
    for i, (hmin, wmin, hmax, wmax) in enumerate(boxes):
        patch_logit = torch.nn.functional.interpolate(
            logits[i : i + 1],
            size=(hmax - hmin, wmax - wmin),
            mode="bilinear",
            align_corners=True,
        )
        logit_patches.append(patch_logit[0, 0].float().cpu().numpy())

    merged = merge_logits(logit_patches, boxes, h, w)
    prob = 1.0 / (1.0 + np.exp(-merged))

    if threshold is None:
        return np.rint(prob * 255).astype(np.uint8)
    return (prob > threshold).astype(np.uint8) * 255


def auto_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    size: int = 1024,
    threshold: float | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    n_h = 2 if h > 768 else 1
    n_w = 2 if w > 768 else 1

    return predict(
        model=model,
        image=image,
        size=size,
        threshold=threshold,
        n=(n_w, n_h),
        overlap=0.1,
    )

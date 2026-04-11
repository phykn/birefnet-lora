import contextlib

import cv2
import numpy as np
import torch

from sdimg.spatial import resize

from ..data.preprocess import preprocess

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    size: int = 1024,
    threshold: float | None = 0.5,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    h, w = image.shape[:2]

    x = preprocess(image)
    x = resize(x, height=size, width=size, interpolation=cv2.INTER_CUBIC)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]
    x = torch.from_numpy(x).to(device)

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
    logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=True
    )
    prob = logits.sigmoid()[0, 0].float().cpu().numpy()

    if threshold is None:
        return prob
    return (prob > threshold).astype(np.uint8) * 255

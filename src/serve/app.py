import asyncio
from typing import Any

import torch
from fastapi import FastAPI

from ..prepare.spec import PreprocessSpec
from .route import router


def read_threshold(model: Any) -> float | None:
    meta = getattr(model, "loaded_meta", None) or {}
    value = meta.get("selection", {}).get("threshold")
    return None if value is None else float(value)


def read_preprocess(model: Any) -> PreprocessSpec:
    return PreprocessSpec.from_meta(getattr(model, "loaded_meta", None))


def build_app(
    model: Any,
    device: torch.device,
    threshold: float | None,
    preprocess: PreprocessSpec | None = None,
) -> FastAPI:
    app = FastAPI(title="BiRefNet-LoRA API")
    app.state.model = model
    app.state.device = device
    app.state.threshold = threshold
    app.state.preprocess = preprocess or PreprocessSpec()
    app.state.predict_sem = asyncio.Semaphore(1)
    app.include_router(router)
    return app

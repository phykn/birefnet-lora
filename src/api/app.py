import asyncio

import torch
from fastapi import FastAPI

from .routes import health

MAX_CONCURRENT_PREDICTIONS = 2


def _register_routes(app: FastAPI) -> None:
    app.include_router(health.router)


def create_app(model, device: torch.device) -> FastAPI:
    app = FastAPI(title="BiRefNet-LoRA Inference API")
    app.state.model = model
    app.state.device = device
    app.state.predict_sem = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
    _register_routes(app)
    return app

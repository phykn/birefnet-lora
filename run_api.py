import argparse
import asyncio
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.adapt.fuse import fuse
from src.build.model import build as build_model
from src.build.model import load as load_model_overlay
from src.serve.route import router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--weight", required=True)
    return parser.parse_args()


def load_model(path: str, device: torch.device):
    cfg = OmegaConf.load("config/model.yaml")
    base = build_model(cfg).to(device)
    model = load_model_overlay(cfg, base, path)
    model.eval()
    return fuse(model)


def read_threshold(model: Any) -> float | None:
    meta = model.loaded_meta or {}
    value = meta.get("selection", {}).get("threshold")
    return None if value is None else float(value)


def build_app(
    model: Any,
    device: torch.device,
    threshold: float | None,
) -> FastAPI:
    app = FastAPI(title="BiRefNet-LoRA API")
    app.state.model = model
    app.state.device = device
    app.state.threshold = threshold
    app.state.predict_sem = asyncio.Semaphore(1)
    app.include_router(router)
    return app


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weight, device)
    app = build_app(
        model=model,
        device=device,
        threshold=read_threshold(model),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

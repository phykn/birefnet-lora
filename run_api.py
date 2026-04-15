import argparse
import asyncio

import torch
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.api.routes import router
from src.ml.build import build_birefnet, build_lora_birefnet_for_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    return parser.parse_args()


def _load_model(weight_path: str, device: torch.device):
    cfg = OmegaConf.load("src/config/model.yaml")
    base = build_birefnet(cfg=cfg).to(device)
    model = build_lora_birefnet_for_inference(
        cfg=cfg, model=base, ckpt_path=weight_path
    )
    model.eval()
    return model


def build_app(model, device: torch.device, max_concurrency: int) -> FastAPI:
    app = FastAPI(title="BiRefNet-LoRA Inference API")
    app.state.model = model
    app.state.device = device
    app.state.predict_sem = asyncio.Semaphore(max_concurrency)
    app.include_router(router)
    return app


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(args.weight, device)
    app = build_app(model=model, device=device, max_concurrency=args.max_concurrency)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

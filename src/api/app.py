import asyncio
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.ml.build import build_birefnet, build_lora_birefnet_for_inference

from .routes import router

MAX_CONCURRENT_PREDICTIONS = 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    weight_path = os.environ["BIREFNET_LORA_WEIGHT"]
    cfg = OmegaConf.load("src/config/model.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    base = build_birefnet(cfg=cfg).to(device)
    model = build_lora_birefnet_for_inference(
        cfg=cfg, model=base, ckpt_path=weight_path
    )
    model.eval()

    app.state.model = model
    app.state.device = device
    app.state.predict_sem = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
    yield


app = FastAPI(title="BiRefNet-LoRA Inference API", lifespan=lifespan)
app.include_router(router)

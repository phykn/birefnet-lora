import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from imstr import decode, encode

from src.ml.inference.predict import predict as run_predict

from .schema import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


def _to_predict_response(arr: np.ndarray) -> PredictResponse:
    height, width = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else None
    return PredictResponse(
        base64_str=encode(arr),
        height=height,
        width=width,
        channels=channels,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", device=request.app.state.device.type)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    try:
        image = decode(body.base64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc

    want_prob_map = body.threshold is None
    threshold = body.threshold

    async with request.app.state.predict_sem:
        result = await run_in_threadpool(
            run_predict, request.app.state.model, image, threshold=threshold
        )
        if request.app.state.device.type == "cuda":
            torch.cuda.empty_cache()

    if want_prob_map:
        result = (result * 255).astype(np.uint8)

    return _to_predict_response(result)

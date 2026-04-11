import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from imstr import decode, encode

from src.ml.inference.predict import predict

from ..schema import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: Request, body: PredictRequest
) -> PredictResponse:
    try:
        image = decode(body.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc

    threshold: float | None = None if body.threshold < 0 else body.threshold

    async with request.app.state.predict_sem:
        result = await run_in_threadpool(
            predict, request.app.state.model, image, threshold=threshold
        )
        if request.app.state.device.type == "cuda":
            torch.cuda.empty_cache()

    if threshold is None:
        result = (result * 255).astype(np.uint8)

    return PredictResponse(mask=encode(result))

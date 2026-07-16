from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..predict.run import predict as predict_mask

from .codec import ImageLimitError, decode, encode
from .schema import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def check_health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", device=request.app.state.device.type)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    threshold = body.threshold
    if threshold is None:
        threshold = request.app.state.threshold
    if threshold is None:
        threshold = 0.5

    async with request.app.state.predict_sem:
        try:
            image = await run_in_threadpool(decode, body.base64_str)
        except ImageLimitError as exc:
            raise HTTPException(status_code=413, detail="image too large") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc

        mask = await run_in_threadpool(
            predict_mask,
            request.app.state.model,
            image,
            output_mode=body.output_mode,
            threshold=threshold,
            tiles=body.tiles,
            overlap=body.overlap,
        )
        data = await run_in_threadpool(encode, mask)

    height, width = mask.shape[:2]
    return PredictResponse(
        id=body.id,
        base64_str=data,
        height=height,
        width=width,
        channel=mask.shape[2] if mask.ndim == 3 else None,
        output_mode=body.output_mode,
        threshold_applied=threshold if body.output_mode == "binary" else None,
    )

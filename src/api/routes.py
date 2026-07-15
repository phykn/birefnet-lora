from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..infer.predict import predict as run_predict

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
        threshold = request.app.state.settings["default_threshold"]

    settings = request.app.state.settings
    async with request.app.state.predict_sem:
        try:
            image = await run_in_threadpool(decode, body.base64_str)
        except ImageLimitError as exc:
            raise HTTPException(status_code=413, detail="image too large") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid image") from exc

        result = await run_in_threadpool(
            run_predict,
            request.app.state.model,
            image,
            output_mode=body.output_mode,
            threshold=threshold,
            size=settings["size"],
            mode=settings["mode"],
            overlap_ratio=settings["overlap_ratio"],
            tile_batch=settings["tile_batch"],
            context_weight=settings["context_weight"],
        )
        encoded = await run_in_threadpool(encode, result)

    height, width = result.shape[:2]
    return PredictResponse(
        id=body.id,
        base64_str=encoded,
        height=height,
        width=width,
        channel=result.shape[2] if result.ndim == 3 else None,
        output_mode=body.output_mode,
        threshold_applied=threshold if body.output_mode == "binary" else None,
    )

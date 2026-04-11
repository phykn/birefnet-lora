from fastapi import APIRouter, Request

from ..schema import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    device = request.app.state.device
    return HealthResponse(status="ok", device=device.type)

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str


class ImageData(BaseModel):
    id: str | None = None
    base64_str: str
    height: int | None
    width: int | None
    channel: int | None


class PredictRequest(ImageData):
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
    )


class PredictResponse(ImageData):
    pass

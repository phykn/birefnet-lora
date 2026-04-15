from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str


class ImageData(BaseModel):
    base64_str: str
    height: int | None
    width: int | None
    channels: int | None


class PredictRequest(ImageData):
    threshold: float = Field(
        0.5,
        ge=-1.0,
        le=1.0,
    )


class PredictResponse(ImageData):
    pass

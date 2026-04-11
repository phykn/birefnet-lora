from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image: str = Field(..., description="imstr-encoded RGB uint8 image")
    threshold: float = Field(
        0.5,
        ge=-1.0,
        le=1.0,
        description="0..1 for binary mask; negative for probability map",
    )


class PredictResponse(BaseModel):
    mask: str = Field(..., description="imstr-encoded uint8 mask or probability map")


class HealthResponse(BaseModel):
    status: str
    device: str

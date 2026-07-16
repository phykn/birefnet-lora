from typing import Literal

from pydantic import BaseModel, Field, PositiveInt

from .codec import MAX_BASE64_LENGTH


class HealthResponse(BaseModel):
    status: str
    device: str


class PredictRequest(BaseModel):
    id: str | None = None
    base64_str: str = Field(min_length=4, max_length=MAX_BASE64_LENGTH)
    output_mode: Literal["binary", "probability"] = "binary"
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    tiles: tuple[PositiveInt, ...] = Field((1,), min_length=1)
    overlap: float = Field(1 / 3, ge=1 / 3, lt=1.0)


class PredictResponse(BaseModel):
    id: str | None
    base64_str: str
    height: int
    width: int
    channel: int | None
    output_mode: Literal["binary", "probability"]
    dtype: Literal["uint8"] = "uint8"
    value_range: tuple[int, int] = (0, 255)
    threshold_applied: float | None

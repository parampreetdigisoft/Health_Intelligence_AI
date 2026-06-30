from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


PillarSignalType = Literal["risk", "trend"]
PillarSignalStatus = Literal["Rising", "Active", "Watch", "Stable", "Critical"]
PillarSignalUrgency = Literal["low", "medium", "high", "critical"]
PillarSignalColor = Literal["green", "yellow", "orange", "red", "blue"]


class PillarLiveSignalCard(BaseModel):
    pillarId: int = Field(..., ge=1)
    type: PillarSignalType
    title: str = Field(..., min_length=3, max_length=55)
    summary: str = Field(..., min_length=10, max_length=100)
    status: PillarSignalStatus
    urgency: PillarSignalUrgency
    color: PillarSignalColor
    sourceUrl: str = Field(..., min_length=12, max_length=2048)

    @field_validator("summary")
    @classmethod
    def trim_summary(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        return cleaned[:100]

    @field_validator("title")
    @classmethod
    def trim_title(cls, value: str) -> str:
        return " ".join(value.split())[:55]

    @field_validator("sourceUrl")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        url = value.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("sourceUrl must start with http:// or https://")
        return url


class PillarLiveSignalsResult(BaseModel):
    updatedAt: str
    headline: str = Field(..., min_length=3, max_length=120)
    subHeadline: str = Field(..., min_length=10, max_length=200)
    pillars: List[PillarLiveSignalCard] = Field(..., min_length=1)

    @field_validator("updatedAt")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class ChatPillarLiveSignalsResponse(BaseModel):
    success: bool
    message: str
    result: PillarLiveSignalsResult

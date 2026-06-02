import re
from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


TrendType = Literal["risk", "trend"]
TrendCategory = Literal[
    "Governance",
    "Conflict",
    "Economy",
    "Climate",
    "Security",
    "Migration",
    "Society",
    "Technology",
    "Health",
]
TrendStatus = Literal["Rising", "Active", "Watch", "Stable", "Critical"]
TrendUrgency = Literal["low", "medium", "high", "critical"]
TrendColor = Literal["green", "yellow", "orange", "red", "blue"]
TrendIcon = Literal[
    "governance",
    "conflict",
    "economy",
    "climate",
    "security",
    "migration",
    "society",
    "technology",
    "health",
]


class EmergingTrendCityCard(BaseModel):
    city: str = Field(..., min_length=2, max_length=200)
    country: str = Field(..., min_length=2, max_length=200)
    cityCode: str = Field(..., min_length=2, max_length=4)
    region: str = Field(..., min_length=2, max_length=80)
    type: TrendType
    title: str = Field(..., min_length=3, max_length=200)
    summary: str = Field(..., min_length=10, max_length=500)
    category: TrendCategory
    status: TrendStatus
    urgency: TrendUrgency
    confidence: int = Field(..., ge=0, le=200)
    icon: TrendIcon
    color: TrendColor
    sourceUrl: str = Field(..., min_length=12, max_length=2048)

    @field_validator("cityCode")
    @classmethod
    def citys_code_uppercase(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("summary")
    @classmethod
    def trim_summary(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        return cleaned[:140]

    @field_validator("sourceUrl")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        url = value.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("sourceUrl must start with http:// or https://")
        if " " in url:
            raise ValueError("sourceUrl must be a single URL with no spaces")
        return url

    @field_validator("title", "summary")
    @classmethod
    def strip_source_mentions(cls, value: str) -> str:
        cleaned = re.sub(
            r"\s*(?:according to|reported by|sources? say|as reported|per reports?).*$",
            "",
            value.strip(),
            flags=re.IGNORECASE,
        )
        return cleaned.strip()


class EmergingTrendsResult(BaseModel):
    updatedAt: str
    headline: str = Field(..., min_length=3, max_length=120)
    subHeadline: str = Field(..., min_length=10, max_length=200)
    cities: List[EmergingTrendCityCard] = Field(..., min_length=1, max_length=250)

    @field_validator("updatedAt")
    @classmethod
    def validate_updated_at(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("updatedAt must be a valid ISO-8601 datetime string") from exc
        return value


class ChatEmergingTrendsResponse(BaseModel):
    success: bool
    message: str
    result: EmergingTrendsResult

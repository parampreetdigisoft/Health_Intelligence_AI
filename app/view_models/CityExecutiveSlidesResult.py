# ============================================================
# RESPONSE MODEL
# ============================================================

from pydantic import BaseModel
from typing import List


class PerformanceSummary(BaseModel):
    trend: str
    summary: str


class CombinedRiskItem(BaseModel):
    rank: int
    title: str
    riskScore: int
    severity: str
    trend: str
    description: str
    recommendation: str


class EarlyWarningItem(BaseModel):
    title: str
    description: str
    timeframe: str
    impactLevel: str


class CityExecutiveSlidesResult(BaseModel):
    cityId: int
    cityName: str

    dailyPerformance: PerformanceSummary

    weeklyPerformance: PerformanceSummary

    monthlyPerformance: PerformanceSummary

    combinedRisks: List[CombinedRiskItem]

    earlyWarnings: List[EarlyWarningItem]


class ChatCityExecutiveSlidesResponse(BaseModel):
    success: bool
    message: str
    result: CityExecutiveSlidesResult
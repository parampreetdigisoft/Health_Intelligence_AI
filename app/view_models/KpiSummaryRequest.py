from pydantic import BaseModel, Field
from typing import List, Optional


class KpiInterpretationBand(BaseModel):
    minRange: Optional[float] = None
    maxRange: Optional[float] = None
    condition: Optional[str] = None
    descriptor: Optional[str] = None
    strategicAction: Optional[str] = None


class KpiSummaryRequest(BaseModel):
    countryName: Optional[str] = None
    layerName: str
    layerCode: str
    purpose: Optional[str] = None
    manualScore: Optional[float] = None
    aiScore: Optional[float] = None
    manualCondition: Optional[str] = None
    aiCondition: Optional[str] = None
    interpretationBands: List[KpiInterpretationBand] = Field(default_factory=list)
    categoryDetails: Optional[str] = None


class KpiSummaryResult(BaseModel):
    summary: str
    scoreInterpretation: Optional[str] = None
    keyTakeaways: List[str] = Field(default_factory=list)
    outlook: Optional[str] = None


class KpiSummaryResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result: Optional[KpiSummaryResult] = None

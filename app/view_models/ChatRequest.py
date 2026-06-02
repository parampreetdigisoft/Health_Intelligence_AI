from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    cityID: int
    questionText: str
    historyText: Optional[str] = None
    pillarID: Optional[int] = None

class ChatGlobalRequest(BaseModel):
    questionText: str
    historyText: Optional[str] = None
    faqid: Optional[int] = None

class ChatCityRequest(BaseModel):
    cityID: int
    questionText: str
    historyText: Optional[str] = None
    faqid: Optional[int] = None
    pillarID: Optional[int] = None

class ChatCityExecutiveSlidesRequest(BaseModel):
    cityId: int

class ChatCityExecutiveSlidesResponse(BaseModel):
    success: bool
    message: str
    result: Any

class ChatCrossComparisionRequest(BaseModel):
    questionText: str
    cityIDs: list[int]
    historyText: Optional[str] = None
    faqid: Optional[int] = None   
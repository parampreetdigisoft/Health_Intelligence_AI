from pydantic import BaseModel
from typing import Any, List, Optional, Dict

class ChatRequest(BaseModel):
    countryID: int
    questionText: str
    historyText: Optional[str] = None
    pillarID: Optional[int] = None

class ChatGlobalRequest(BaseModel):
    questionText: str
    historyText: Optional[str] = None
    faqid: Optional[int] = None

class ChatCountryRequest(BaseModel):
    countryID: int
    questionText: str
    historyText: Optional[str] = None
    faqid: Optional[int] = None
    pillarID: Optional[int] = None


class ChatCrossComparisionRequest(BaseModel):
    questionText: str
    countryIDs: list[int]
    historyText: Optional[str] = None
    faqid: Optional[int] = None


class ChatCountryExecutiveSlidesRequest(BaseModel):
    countryId: int

class ChatCountryExecutiveSlidesResponse(BaseModel):
    success: bool
    message: str
    result: Any
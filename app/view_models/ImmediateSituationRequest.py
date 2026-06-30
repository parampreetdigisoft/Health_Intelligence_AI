from pydantic import BaseModel
from typing import List, Optional, Dict

class ImmediateSituationRequest(BaseModel):
    country_id: int
    countryName: str
    continent: str

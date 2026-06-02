from pydantic import BaseModel
from typing import List, Optional, Dict

class ImmediateSituationRequest(BaseModel):
    city_id: int
    cityName: str
    country: str

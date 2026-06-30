from typing import Optional
from pydantic import BaseModel

class MissingPillarQuestionRequest(BaseModel):
    countryID: int
    pillarID: Optional[int] = None
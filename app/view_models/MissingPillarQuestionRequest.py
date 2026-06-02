from typing import Optional
from pydantic import BaseModel

class MissingPillarQuestionRequest(BaseModel):
    cityID: int
    pillarID: Optional[int] = None
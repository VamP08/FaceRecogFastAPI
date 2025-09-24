# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

# --- API Response Models ---

class StandardResponse(BaseModel):
    STATUS: int
    CODE: int
    FLAG: bool
    MESSAGE: str
    DATA: Optional[dict] = None

class FaceResult(BaseModel):
    name: str
    box: List[int]
    score: float

class RecognitionResponse(BaseModel):
    faces: List[FaceResult]
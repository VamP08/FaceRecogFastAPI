# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

# --- API Response Models ---

class EmployeeInfo(BaseModel):
    id: str
    name: str
    member_code: Optional[str] = None

    class Config:
        from_attributes = True

class EmployeeListResponse(BaseModel):
    employees: List[EmployeeInfo]

class StandardResponse(BaseModel):
    STATUS: int
    CODE: int
    FLAG: bool
    MESSAGE: str
    DATA: Optional[dict] = None

class FaceResult(BaseModel):
    name: str
    member_code: Optional[str] = None
    box: List[int]
    score: float

class RecognitionResponse(BaseModel):
    faces: List[FaceResult]
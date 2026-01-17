from pydantic import BaseModel
from typing import Optional, Literal

class CreateJobResponse(BaseModel):
    reference_id: str
    status: Literal["pending", "processing"]
    message: str = "Face-swap job accepted"

class JobStatusResponse(BaseModel):
    reference_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    result_image_url: Optional[str] = None
    processing_ms: Optional[int] = None
    error: Optional[str] = None

"""Request/response models for the FastAPI endpoints."""
from typing import Optional, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str  # "approve" or "reject"


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    interrupted: bool
    interrupt_data: Optional[dict[str, Any]] = None
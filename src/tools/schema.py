from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    reset: bool = False


class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    assistant_message: str
    should_exit: bool = False


class LoginRequest(BaseModel):
    account: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    user_id: str
    account: str
    created: bool


class SignupRequest(BaseModel):
    account: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)


class SignupResponse(BaseModel):
    user_id: str
    account: str

class UpdateThreadFieldsInput(BaseModel):
    user_id: str = Field(..., description="User id")
    thread_id: str = Field(..., description="Thread id")
    fields_json: Optional[str] = Field(None, description="JSON string of fields to $set into thread document")
    fields: Optional[Dict[str, Any]] = Field(None, description="Dict of fields to $set into thread document")

class ValidateIntentRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ValidateIntentResponse(BaseModel):
    user_id: str
    should_start: bool
    confidence: float
    reason: str
    normalized_message: str

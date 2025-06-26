from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class SessionStatus(BaseModel):
    valid: bool
    message: Optional[str] = None
    questions_loaded: Optional[bool] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class QuestionResponse(BaseModel):
    status: str
    count: Optional[int] = None
    first_question: Optional[str] = None
    session_token: Optional[str] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# Tutti i modelli sono ora centralizzati in Main.models
from Main.models import InterviewResponse, ErrorResponse, QuestionResponse, Token, TokenData


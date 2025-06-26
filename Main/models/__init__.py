# In questo file implementiamo direttamente tutti i modelli per evitare problemi di importazione circolare

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from datetime import datetime

# ------------------------------------------------------------------------------
# Modelli di autenticazione
# ------------------------------------------------------------------------------

class TokenData(BaseModel):
    username: str
    exp: Optional[datetime] = None


class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True


# ------------------------------------------------------------------------------
# Modelli di risposta generici
# ------------------------------------------------------------------------------

class BaseResponse(BaseModel):
    status: str
    message: str


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


# ------------------------------------------------------------------------------
# Modelli per le domande
# ------------------------------------------------------------------------------

class QuestionBase(BaseModel):
    id: str
    text: str
    category: Optional[str] = None
    difficulty: Optional[str] = None


class QuestionWithMetadata(QuestionBase):
    topic: Optional[str] = None
    subtopics: List[str] = []
    keywords: Dict[str, Any] = {}


class QuestionResponse(BaseResponse):
    question: QuestionWithMetadata


class QuestionListResponse(BaseResponse):
    questions: List[QuestionWithMetadata]


# ------------------------------------------------------------------------------
# Modelli per le interviste
# ------------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    answer_text: str


class InterviewState(BaseModel):
    interview_id: str
    user_id: str
    start_time: datetime
    current_question_id: Optional[str] = None
    questions_asked: List[str] = []
    answers: Dict[str, str] = {}
    completed: bool = False
    score: Optional[int] = None


class InterviewResponse(BaseResponse):
    interview_id: str


class InterviewStatusResponse(InterviewResponse):
    user_id: str
    start_time: datetime
    current_question_id: Optional[str]
    questions_asked: List[str]
    answers_count: int
    completed: bool
    score: Optional[int]


class InterviewQuestionResponse(InterviewResponse):
    question: QuestionBase


class InterviewResultResponse(InterviewResponse):
    score: int
    questions_asked: int
    answers_provided: int


# ------------------------------------------------------------------------------
# Modelli per TTS
# ------------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None


class TTSResponse(BaseResponse):
    audio_base64: str

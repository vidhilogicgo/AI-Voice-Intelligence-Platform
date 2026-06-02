from enum import Enum

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TranscriptSegment(BaseModel):
    segment_id: int = 0
    speaker: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    start_time: str
    end_time: str
    text: str


class Summary(BaseModel):
    short: str
    detailed: str
    key_points: list[str] = Field(default_factory=list)


class Insights(BaseModel):
    pain_points: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    feature_requests: list[str] = Field(default_factory=list)
    sentiment: str
    buying_intent: str
    action_items: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    transcript: list[TranscriptSegment]
    summary: Summary
    insights: Insights


class UploadResponse(BaseModel):
    audio_id: str
    filename: str
    status: ProcessingStatus


class StatusResponse(BaseModel):
    audio_id: str
    status: ProcessingStatus
    state: str | None = None


class ResultResponse(BaseModel):
    audio_id: str
    status: ProcessingStatus
    result: AnalysisResult


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class AskResponse(BaseModel):
    audio_id: str
    question: str
    answer: str
    sources: list[TranscriptSegment] = Field(default_factory=list)

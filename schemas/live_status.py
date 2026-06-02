from pydantic import BaseModel


class LiveStatusResponse(BaseModel):
    success_calls: int
    failed_calls: int
    total_calls: int
    success_rate: float
    failure_rate: float
    avg_response_time_ms: float

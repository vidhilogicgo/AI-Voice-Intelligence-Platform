from fastapi import APIRouter

from schemas.common import ApiResponse
from schemas.health import HealthResponse
from schemas.live_status import LiveStatusResponse
from services.job_store import job_store

router = APIRouter()


@router.get("/api/health", response_model=ApiResponse[HealthResponse])
async def health_check() -> ApiResponse[HealthResponse]:
    return ApiResponse(
        success=True,
        message="Service is healthy.",
        data=HealthResponse(status="ok", service="ai-voice-intelligence-api"),
    )


@router.get("/api/live-status", response_model=ApiResponse[LiveStatusResponse])
async def live_status() -> ApiResponse[LiveStatusResponse]:
    return ApiResponse(
        success=True,
        message="Live status fetched successfully.",
        data=LiveStatusResponse(**job_store.get_live_status()),
    )

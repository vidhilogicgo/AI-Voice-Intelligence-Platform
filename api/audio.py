from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from starlette import status

from core.config import get_settings
from core.errors import AppError
from schemas.audio import (
    AskRequest,
    AskResponse,
    ProcessingStatus,
    ResultResponse,
    StatusResponse,
    UploadResponse,
)
from schemas.common import ApiResponse
from services.audio_handler import AudioHandler
from services.job_store import AudioJob, job_store
from services.pipeline import pipeline
from services.qa import TranscriptQAService

router = APIRouter()


@router.post(
    "/api/audio/upload",
    response_model=ApiResponse[UploadResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> ApiResponse[UploadResponse]:
    settings = get_settings()
    file_id = str(uuid4())
    audio_handler = AudioHandler(settings)
    try:
        file_path = await audio_handler.save_upload(file, file_id)
    except AppError as exc:
        job_store.record_processing_log(
            audio_id=file_id,
            status="failure",
            stage="upload_failed",
            message=exc.message,
            details={
                "error_code": exc.code,
                "filename": file.filename,
            },
        )
        raise

    job = job_store.create(
        AudioJob(
            id=file_id,
            filename=file.filename or file_path.name,
            file_path=file_path,
            status=ProcessingStatus.pending,
            message="Audio upload accepted and pending processing.",
        )
    )
    job_store.record_processing_log(
        audio_id=job.id,
        status="success",
        stage="upload",
        message="Audio upload accepted and pending processing.",
        details={
            "filename": job.filename,
        },
    )
    background_tasks.add_task(pipeline.process, job)

    return ApiResponse(
        success=True,
        message=job.message or "Audio pending processing.",
        data=UploadResponse(
            audio_id=job.id,
            filename=job.filename,
            status=job.status,
        ),
    )


@router.get("/api/audio/{id}/status", response_model=ApiResponse[StatusResponse])
async def get_processing_status(id: str) -> ApiResponse[StatusResponse]:
    job = _get_job_or_404(id)
    return ApiResponse(
        success=True,
        message="Processing status fetched successfully.",
        data=StatusResponse(
            audio_id=job.id,
            status=job.status,
        ),
    )


@router.get("/api/audio/{id}/result", response_model=ApiResponse[ResultResponse])
async def get_analysis_result(id: str) -> ApiResponse[ResultResponse]:
    job = _get_job_or_404(id)
    if job.status != ProcessingStatus.completed or job.result is None:
        raise AppError(
            "Analysis result is not ready yet.",
            status_code=status.HTTP_409_CONFLICT,
            code="result_not_ready",
        )

    return ApiResponse(
        success=True,
        message="Analysis result fetched successfully.",
        data=ResultResponse(audio_id=job.id, status=job.status, result=job.result),
    )


@router.post("/api/audio/{id}/ask", response_model=ApiResponse[AskResponse])
async def ask_question(id: str, payload: AskRequest) -> ApiResponse[AskResponse]:
    job = _get_job_or_404(id)
    if job.status != ProcessingStatus.completed or job.result is None:
        raise AppError(
            "Q&A is available after analysis completes.",
            status_code=status.HTTP_409_CONFLICT,
            code="analysis_not_ready",
        )

    qa_service = TranscriptQAService()
    answer = await qa_service.answer(
        audio_id=job.id,
        question=payload.question,
        transcript=job.result.transcript,
    )
    job_store.record_qa(answer)
    return ApiResponse(
        success=True,
        message="Question answered successfully.",
        data=answer,
    )


def _get_job_or_404(audio_id: str) -> AudioJob:
    job = job_store.get(audio_id)
    if job is None:
        raise AppError(
            "Audio job was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            code="audio_not_found",
        )
    return job

from pathlib import Path

from core.config import get_settings
from core.errors import AppError
from schemas.audio import AnalysisResult, ProcessingStatus
from services.audio_preprocessor import AudioPreprocessor
from services.diarization import SpeakerDiarizationService
from services.insight_extraction import InsightExtractionService
from services.job_store import AudioJob, job_store
from services.summarization import SummarizationService
from services.transcription import TranscriptionService


class VoiceIntelligencePipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.audio_preprocessor = AudioPreprocessor(self.settings)
        self.transcription = TranscriptionService(self.settings)
        self.diarization = SpeakerDiarizationService(self.settings)
        self.summarization = SummarizationService(self.settings)
        self.insights = InsightExtractionService(self.settings)

    async def process(self, job: AudioJob) -> AudioJob:
        _set_processing_stage(job, "processing_started", "Audio analysis started.")

        try:
            _set_processing_stage(job, "preprocessing", "Preparing audio for AI processing.")
            processed_path = await self.audio_preprocessor.prepare_for_ai(
                input_path=job.file_path,
                audio_id=job.id,
            )
            job.processed_file_path = processed_path
            job_store.update(job)
            job_store.record_processing_log(
                audio_id=job.id,
                status="success",
                stage="audio_preprocessing",
                message="Audio preprocessing completed.",
            )
            print("✅ [PREPROCESSING] Audio preprocessing model/tool (FFmpeg) succeeded.")

            _set_processing_stage(job, "transcription", "Transcribing audio.")
            timestamped_transcript = await self.transcription.transcribe(processed_path)
            job_store.record_processing_log(
                audio_id=job.id,
                status="success",
                stage="transcription",
                message="Audio transcription completed.",
                details={"segments": len(timestamped_transcript)},
            )
            _set_processing_stage(job, "diarization", "Identifying speakers.")
            transcript = await self.diarization.diarize(
                timestamped_transcript,
                audio_path=processed_path,
            )
            job_store.record_processing_log(
                audio_id=job.id,
                status="success",
                stage="diarization",
                message="Speaker diarization completed.",
                details={"segments": len(transcript)},
            )
            _set_processing_stage(job, "summarization", "Generating summary.")
            summary = await self.summarization.summarize(transcript)
            job_store.record_processing_log(
                audio_id=job.id,
                status="success",
                stage="summarization",
                message="Summary generation completed.",
            )
            _set_processing_stage(job, "insight_extraction", "Extracting business insights.")
            insights = await self.insights.extract(transcript)
            job_store.record_processing_log(
                audio_id=job.id,
                status="success",
                stage="insight_extraction",
                message="Business insight extraction completed.",
            )

            job.result = AnalysisResult(
                transcript=transcript,
                summary=summary,
                insights=insights,
            )
            job.status = ProcessingStatus.completed
            job.stage = "completed"
            job.message = "Audio analysis completed."
            job.error = None
            print("🎉 [PIPELINE] Entire audio processing pipeline completed successfully!")
        except AppError as exc:
            print(f"❌ [{job.stage.upper() if job.stage else 'PIPELINE'}] Step failed: {exc.message}")
            _mark_failed(
                job,
                stage=job.stage or "processing",
                user_message=_safe_stage_error_message(job.stage, exc.message),
                details={
                    "error_code": exc.code,
                    "exception_type": exc.__class__.__name__,
                    "debug_message": exc.message[:500],
                },
            )
        except Exception as exc:
            print(f"❌ [{job.stage.upper() if job.stage else 'PIPELINE'}] Step failed with unexpected error: {exc}")
            _mark_failed(
                job,
                stage=job.stage or "processing",
                user_message=_safe_stage_error_message(job.stage),
                details={
                    "exception_type": exc.__class__.__name__,
                    "debug_message": str(exc)[:500],
                },
            )
        finally:
            _safe_delete_audio_file(job.file_path)
            if job.processed_file_path and job.processed_file_path != job.file_path:
                _safe_delete_audio_file(job.processed_file_path)

        updated_job = job_store.update(job)
        if updated_job.status == ProcessingStatus.completed:
            job_store.record_processing_log(
                audio_id=updated_job.id,
                status="success",
                stage="processing_completed",
                message=updated_job.message or "Audio analysis completed.",
            )
        return updated_job


def _set_processing_stage(job: AudioJob, stage: str, message: str) -> None:
    job.status = ProcessingStatus.processing
    job.stage = stage
    job.message = message
    job.error = None
    job_store.update(job)
    job_store.record_processing_log(
        audio_id=job.id,
        status="success",
        stage=stage,
        message=message,
    )


def _mark_failed(
    job: AudioJob,
    *,
    stage: str,
    user_message: str,
    details: dict[str, str],
) -> None:
    job.status = ProcessingStatus.failed
    job.stage = f"{stage}_failed" if not stage.endswith("_failed") else stage
    job.message = user_message
    job.error = user_message
    job_store.record_processing_log(
        audio_id=job.id,
        status="failure",
        stage=job.stage,
        message=user_message,
        details=details,
    )


def _safe_stage_error_message(
    stage: str | None,
    app_message: str | None = None,
) -> str:
    if stage == "preprocessing":
        return "Audio preprocessing failed. Please upload a valid audio file."
    if stage == "transcription":
        return "Audio transcription failed. Please try another audio file."
    if stage == "diarization":
        return "Speaker diarization failed."
    if stage in {"summarization", "insight_extraction"}:
        return "AI processing failed while analyzing the transcript."
    if app_message:
        return app_message
    return "Audio analysis failed."


def _safe_delete_audio_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[PIPELINE] Warning: Could not delete temporary audio file {path}: {exc}")


pipeline = VoiceIntelligencePipeline()

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import get_settings

from schemas.audio import AnalysisResult, AskResponse, ProcessingStatus


@dataclass
class AudioJob:
    id: str
    filename: str
    file_path: Path
    status: ProcessingStatus
    message: str | None = None
    stage: str | None = None
    error: str | None = None
    result: AnalysisResult | None = None
    processed_file_path: Path | None = None


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, AudioJob] = {}

    def create(self, job: AudioJob) -> AudioJob:
        self._jobs[job.id] = job
        return job

    def get(self, audio_id: str) -> AudioJob | None:
        return self._jobs.get(audio_id)

    def update(self, job: AudioJob) -> AudioJob:
        self._jobs[job.id] = job
        return job

    def record_qa(self, response: AskResponse) -> None:
        return None

    def record_processing_log(
        self,
        *,
        audio_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        return None

    def get_live_status(self) -> dict[str, float | int]:
        success_calls = sum(
            1 for job in self._jobs.values() if job.status == ProcessingStatus.completed
        )
        failed_calls = sum(
            1 for job in self._jobs.values() if job.status == ProcessingStatus.failed
        )
        return _build_live_status(
            success_calls=success_calls,
            failed_calls=failed_calls,
            response_times=[],
        )


class MongoJobStore(InMemoryJobStore):
    def __init__(self, uri: str, db_name: str) -> None:
        super().__init__()
        import certifi
        from pymongo import MongoClient

        self._client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            tlsCAFile=certifi.where(),
        )
        self._client.admin.command("ping")
        self._db = self._client[db_name]
        self._ensure_indexes()

    def create(self, job: AudioJob) -> AudioJob:
        super().create(job)
        now = _utc_now()
        self._db.audio_files.update_one(
            {"_id": job.id},
            {
                "$setOnInsert": {
                    "_id": job.id,
                    "audio_id": job.id,
                    "file_type": job.file_path.suffix.lower().lstrip("."),
                    "filename": job.filename,
                    "created_at": now,
                },
                "$set": {
                    "status": job.status.value,
                    "error": job.error,
                    "updated_at": now,
                },
            },
            upsert=True,
        )
        return job

    def get(self, audio_id: str) -> AudioJob | None:
        audio_document = self._db.audio_files.find_one({"audio_id": audio_id})
        if audio_document is None:
            return None

        job = AudioJob(
            id=audio_id,
            filename=str(
                audio_document.get("filename")
                or audio_id
            ),
            file_path=Path(str(audio_document.get("file_path") or "")),
            processed_file_path=_optional_path(audio_document.get("processed_file_path")),
            status=ProcessingStatus(
                audio_document.get("status", ProcessingStatus.pending.value)
            ),
            message=None,
            error=audio_document.get("error"),
            result=self._load_result(audio_id),
        )
        return super().create(job)

    def update(self, job: AudioJob) -> AudioJob:
        super().update(job)
        now = _utc_now()
        self._db.audio_files.update_one(
            {"_id": job.id},
            {
                "$setOnInsert": {
                    "_id": job.id,
                    "audio_id": job.id,
                    "file_type": job.file_path.suffix.lower().lstrip("."),
                    "created_at": now,
                },
                "$set": _audio_file_status_fields(job, now),
            },
            upsert=True,
        )
        if job.result is not None:
            self._upsert_analysis_result(job)
        return job

    def record_qa(self, response: AskResponse) -> None:
        try:
            self._db.qa_history.insert_one(
                {
                    "audio_id": response.audio_id,
                    "question": response.question,
                    "answer": response.answer,
                    "source_segment_ids": [
                        segment.segment_id for segment in response.sources
                    ],
                    "sources": [_model_to_dict(segment) for segment in response.sources],
                    "created_at": _utc_now(),
                }
            )
        except Exception as exc:
            print(f"[JOB_STORE] Warning: Could not save Q&A history: {exc}")

    def record_processing_log(
        self,
        *,
        audio_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            document = {
                "audio_id": audio_id,
                "status": status,
                "details": details or {},
                "created_at": _utc_now(),
            }
            if status == "failure" and message:
                document["error"] = message

            self._db.processing_logs.insert_one(
                document
            )
        except Exception as exc:
            print(f"[JOB_STORE] Warning: Could not save processing log: {exc}")

    def get_live_status(self) -> dict[str, float | int]:
        success_calls = self._db.audio_files.count_documents(
            {"status": ProcessingStatus.completed.value}
        )
        failed_calls = self._db.audio_files.count_documents(
            {"status": ProcessingStatus.failed.value}
        )
        response_times = [
            _response_time_seconds(document)
            for document in self._db.audio_files.find(
                {
                    "status": {
                        "$in": [
                            ProcessingStatus.completed.value,
                            ProcessingStatus.failed.value,
                        ]
                    },
                    "started_at": {"$type": "date"},
                    "completed_at": {"$type": "date"},
                },
                {"started_at": 1, "completed_at": 1},
            )
        ]
        live_status = _build_live_status(
            success_calls=success_calls,
            failed_calls=failed_calls,
            response_times=[
                seconds for seconds in response_times if seconds is not None
            ],
        )
        self._upsert_live_status(live_status)
        return live_status

    def _ensure_indexes(self) -> None:
        self._db.audio_files.create_index("audio_id", unique=True)
        self._db.results.create_index("audio_id", unique=True)
        self._db.qa_history.create_index("audio_id")
        self._db.qa_history.create_index([("audio_id", 1), ("created_at", -1)])
        self._db.model_usage_logs.create_index("audio_id")
        self._db.processing_logs.create_index("audio_id")
        self._db.live_status.create_index("key", unique=True)
        self._db.live_status.create_index("updated_at")

    def _upsert_live_status(self, live_status: dict[str, float | int]) -> None:
        now = _utc_now()
        self._db.live_status.update_one(
            {"key": "live_status"},
            {
                "$set": {
                    **live_status,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "key": "live_status",
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def _upsert_analysis_result(self, job: AudioJob) -> None:
        if job.result is None:
            return

        now = _utc_now()
        transcript_text = "\n".join(segment.text for segment in job.result.transcript)
        speakers = sorted({segment.speaker for segment in job.result.transcript})
        segments = [_model_to_dict(segment) for segment in job.result.transcript]
        self._db.results.update_one(
            {"_id": job.id},
            {
                "$set": {
                    "audio_id": job.id,
                    "transcript": {
                        "full_text": transcript_text,
                        "segments": segments,
                        "speakers": speakers,
                        "speakers_count": len(speakers),
                        "segment_count": len(segments),
                    },
                    "summary": _model_to_dict(job.result.summary),
                    "insights": _model_to_dict(job.result.insights),
                    "updated_at": now,
                },
                "$setOnInsert": {"_id": job.id, "created_at": now},
            },
            upsert=True,
        )
        self._db.transcript_segments.delete_many({"audio_id": job.id})

    def _load_result(self, audio_id: str) -> AnalysisResult | None:
        result = self._db.results.find_one({"audio_id": audio_id})
        if result is not None:
            transcript = result.get("transcript")
            summary = result.get("summary")
            insights = result.get("insights")
            if (
                isinstance(transcript, dict)
                and isinstance(summary, dict)
                and isinstance(insights, dict)
            ):
                segments = transcript.get("segments")
                if isinstance(segments, list) and segments:
                    return AnalysisResult(
                        transcript=[
                            _strip_mongo_fields(segment)
                            for segment in segments
                            if isinstance(segment, dict)
                        ],
                        summary=_strip_mongo_fields(summary),
                        insights=_strip_mongo_fields(insights),
                    )

        transcript = self._db.transcripts.find_one({"audio_id": audio_id})
        summary = self._db.summaries.find_one({"audio_id": audio_id})
        insights = self._db.insights.find_one({"audio_id": audio_id})
        if summary is None or insights is None or transcript is None:
            return None

        segments = transcript.get("segments")
        if not isinstance(segments, list) or not segments:
            segments = [
                _strip_mongo_fields(segment)
                for segment in self._db.transcript_segments.find(
                    {"audio_id": audio_id},
                    sort=[("segment_id", 1)],
                )
            ]
        if not segments:
            return None

        return AnalysisResult(
            transcript=[_strip_mongo_fields(segment) for segment in segments],
            summary=_strip_mongo_fields(summary),
            insights=_strip_mongo_fields(insights),
        )


def _create_job_store() -> InMemoryJobStore:
    settings = get_settings()
    if not settings.mongodb_uri:
        return InMemoryJobStore()

    try:
        return MongoJobStore(settings.mongodb_uri, settings.mongodb_db_name)
    except Exception as exc:
        print(f"[JOB_STORE] MongoDB unavailable, using in-memory store: {exc}")
        return InMemoryJobStore()


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _strip_mongo_fields(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in {"_id", "audio_id", "created_at", "updated_at"}
    }


def _optional_path(value: Any) -> Path | None:
    if value is None or not str(value):
        return None
    return Path(str(value))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _audio_file_status_fields(job: AudioJob, now: datetime) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "filename": job.filename,
        "status": job.status.value,
        "error": job.error,
        "updated_at": now,
    }
    if job.status == ProcessingStatus.processing:
        fields["started_at"] = now
    if job.status in {ProcessingStatus.completed, ProcessingStatus.failed}:
        fields["completed_at"] = now
    return fields


def _response_time_seconds(document: dict[str, Any]) -> float | None:
    started_at = document.get("started_at")
    completed_at = document.get("completed_at")
    if not isinstance(started_at, datetime) or not isinstance(completed_at, datetime):
        return None
    return max(0.0, (completed_at - started_at).total_seconds())


def _build_live_status(
    *,
    success_calls: int,
    failed_calls: int,
    response_times: list[float],
) -> dict[str, float | int]:
    total_calls = success_calls + failed_calls
    success_rate = (
        round((success_calls / total_calls) * 100, 2)
        if total_calls
        else 0.0
    )
    failure_rate = (
        round((failed_calls / total_calls) * 100, 2)
        if total_calls
        else 0.0
    )
    avg_response_time_ms = (
        round((sum(response_times) / len(response_times)) * 1000, 2)
        if response_times
        else 0.0
    )
    return {
        "success_calls": success_calls,
        "failed_calls": failed_calls,
        "total_calls": total_calls,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "avg_response_time_ms": avg_response_time_ms,
    }


job_store = _create_job_store()

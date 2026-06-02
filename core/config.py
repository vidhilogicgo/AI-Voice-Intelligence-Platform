from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Voice Intelligence API"
    debug: bool = False
    upload_dir: Path = Path("storage/uploads")
    processed_audio_dir: Path = Path("storage/processed")
    max_upload_mb: int = 150
    allowed_audio_extensions: set[str] = field(
        default_factory=lambda: {".mp3", ".wav", ".m4a"}
    )
    transcription_sample_rate: int = 16000
    transcription_channels: int = 1
    transcription_engine: str = "faster-whisper"
    transcription_model: str = "base"
    transcription_device: str = "cpu"
    transcription_compute_type: str = "int8"
    transcription_language: str | None = None
    diarization_engine: str = "heuristic"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_auth_token: str | None = None
    diarization_default_speakers: int = 2
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 10
    diarization_clustering_threshold: float = 0.7
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    assemblyai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_summary_model: str = "llama-3.1-8b-instant"
    groq_insight_model: str = "llama-3.1-8b-instant"
    groq_qa_model: str = "llama-3.1-8b-instant"
    qa_chunk_max_chars: int = 1800
    qa_chunk_overlap_segments: int = 1
    qa_retrieval_top_k: int = 4
    qa_retrieval_engine: str = "faiss"
    qa_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mongodb_uri: str | None = None
    mongodb_db_name: str = "ai_voice_intelligence"


def get_settings() -> Settings:
    _load_env_file()
    groq_model = "llama-3.1-8b-instant"
    return Settings(
        app_name="AI Voice Intelligence API",
        debug=False,
        upload_dir=Path("storage/uploads"),
        processed_audio_dir=Path("storage/processed"),
        max_upload_mb=150,
        transcription_sample_rate=16000,
        transcription_channels=1,
        transcription_engine="faster-whisper",
        transcription_model="base",
        transcription_device="cpu",
        transcription_compute_type="int8",
        transcription_language=None,
        diarization_engine="pyannote" if _diarization_auth_token() else "heuristic",
        diarization_model="pyannote/speaker-diarization-3.1",
        diarization_auth_token=_diarization_auth_token(),
        diarization_default_speakers=2,
        diarization_min_speakers=1,
        diarization_max_speakers=10,
        diarization_clustering_threshold=0.7,
        cors_origins=["*"],
        assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=groq_model,
        groq_summary_model=groq_model,
        groq_insight_model=groq_model,
        groq_qa_model=groq_model,
        qa_chunk_max_chars=1800,
        qa_chunk_overlap_segments=1,
        qa_retrieval_top_k=4,
        qa_retrieval_engine="faiss",
        qa_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        mongodb_uri=_optional_env("MONGODB_URI"),
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "ai_voice_intelligence"),
    )


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _diarization_auth_token() -> str | None:
    return (
        _optional_env("HUGGINGFACE_TOKEN")
        or _optional_env("HF_TOKEN")
        or _optional_env("PYANNOTE_AUTH_TOKEN")
    )


def _load_env_file() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue

        key, value = stripped_line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

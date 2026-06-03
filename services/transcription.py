import asyncio
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
import re
import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from starlette import status

from core.config import Settings
from core.errors import AppError
from core.usage_logging import log_model_usage
from schemas.audio import TranscriptSegment

MAX_MERGED_SEGMENT_CHARS = 650
MAX_MERGE_GAP_SECONDS = 1.25
MIN_USEFUL_TEXT_CHARS = 2
NOISE_ONLY_PATTERNS = (
    r"^\[(music|applause|laughter|noise|silence|inaudible|background noise)\]$",
    r"^\((music|applause|laughter|noise|silence|inaudible|background noise)\)$",
    r"^(music|applause|laughter|noise|silence)$",
)
COMMON_ASR_ARTIFACTS = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to our channel",
}


class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    async def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise AppError("Audio file was not found.", code="audio_not_found")

        # Try AssemblyAI first if API key is available
        if self.settings.assemblyai_api_key:
            try:
                print("\n🌐 [TRANSCRIPTION] Attempting AssemblyAI API transcription...")
                result = await asyncio.to_thread(
                    self._transcribe_with_assemblyai,
                    audio_path,
                )
                print("✅ [TRANSCRIPTION] AssemblyAI transcription completed successfully!")
                return result
            except Exception as exc:
                print(f"⚠️  [TRANSCRIPTION] AssemblyAI failed: {exc}")
                print(f"📍 [TRANSCRIPTION] Falling back to local Whisper ({self.settings.transcription_model}) on {self.settings.transcription_device.upper()}")

        # Fallback to local transcription
        engine = self.settings.transcription_engine.strip().lower()
        print(f"\n🖥️  [TRANSCRIPTION] Using local {engine} ({self.settings.transcription_model} model) on {self.settings.transcription_device.upper()}")
        
        if engine in {"faster-whisper", "faster_whisper"}:
            log_model_usage(
                provider="local",
                model=self.settings.transcription_model,
                purpose="speech-to-text transcription",
                details=(
                    f"engine=faster-whisper device={self.settings.transcription_device} "
                    f"compute_type={self.settings.transcription_compute_type}"
                ),
            )
            try:
                result = await asyncio.to_thread(
                    self._transcribe_with_faster_whisper,
                    audio_path,
                )
                print(f"✅ [TRANSCRIPTION] Local faster-whisper ({self.settings.transcription_model}) transcription succeeded.")
                return result
            except Exception as exc:
                print(f"❌ [TRANSCRIPTION] Local faster-whisper ({self.settings.transcription_model}) transcription failed: {exc}")
                if isinstance(exc, AppError):
                    raise
                raise AppError(
                    "Local transcription failed. Check the audio file and transcription settings.",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="local_transcription_failed",
                ) from exc
        if engine in {"whisper", "openai-whisper", "openai_whisper"}:
            log_model_usage(
                provider="local",
                model=self.settings.transcription_model,
                purpose="speech-to-text transcription",
                details=f"engine=openai-whisper device={self.settings.transcription_device}",
            )
            try:
                result = await asyncio.to_thread(
                    self._transcribe_with_openai_whisper,
                    audio_path,
                )
                print(f"✅ [TRANSCRIPTION] Local openai-whisper ({self.settings.transcription_model}) transcription succeeded.")
                return result
            except Exception as exc:
                print(f"❌ [TRANSCRIPTION] Local openai-whisper ({self.settings.transcription_model}) transcription failed: {exc}")
                if isinstance(exc, AppError):
                    raise
                raise AppError(
                    "Local transcription failed. Check the audio file and transcription settings.",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    code="local_transcription_failed",
                ) from exc

        raise AppError(
            "Unsupported transcription engine. Use 'faster-whisper' or 'whisper'.",
            code="unsupported_transcription_engine",
        )

    def _transcribe_with_assemblyai(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe audio using AssemblyAI cloud API."""
        try:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
        except IOError as exc:
            raise AppError(
                f"Failed to read audio file: {exc}",
                code="audio_read_error",
            ) from exc

        # Upload audio to AssemblyAI
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {
            "Authorization": self.settings.assemblyai_api_key,
        }
        
        try:
            print(f"  → Uploading audio file ({len(audio_data) / 1024 / 1024:.1f}MB)...")
            upload_request = Request(
                upload_url,
                data=audio_data,
                headers=headers,
                method="POST",
            )
            with urlopen(upload_request, timeout=30) as response:
                upload_result = json.loads(response.read())
                audio_url = upload_result.get("upload_url")
                if not audio_url:
                    raise AppError(
                        "AssemblyAI upload failed: missing upload URL in response.",
                        code="assemblyai_upload_failed",
                    )
        except HTTPError as exc:
            raise AppError(
                f"AssemblyAI upload failed: {_format_http_error(exc)}",
                code="assemblyai_upload_failed",
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AppError(
                f"AssemblyAI upload failed: {exc}",
                code="assemblyai_upload_failed",
            ) from exc
        except json.JSONDecodeError as exc:
            raise AppError(
                "AssemblyAI upload returned an invalid response.",
                code="assemblyai_invalid_response",
            ) from exc

        # Request transcription with speaker labels for diarization
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        transcript_data = {
            "audio_url": audio_url,
            "speech_models": ["universal-3-pro"],  # Latest and most accurate
            "speaker_labels": True,  # Enable speaker diarization
        }
        
        # Only add language if explicitly set (don't add if using default)
        if self.settings.transcription_language:
            transcript_data["language_code"] = self.settings.transcription_language
        
        try:
            transcript_request = Request(
                transcript_url,
                data=json.dumps(transcript_data).encode(),
                headers={
                    **headers,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(transcript_request, timeout=30) as response:
                transcript_response = json.loads(response.read())
                transcript_id = transcript_response.get("id")
                if not transcript_id:
                    raise AppError(
                        "AssemblyAI transcription request failed: missing transcript ID in response.",
                        code="assemblyai_request_failed",
                    )
        except HTTPError as exc:
            raise AppError(
                f"AssemblyAI transcription request failed: {_format_http_error(exc)}",
                code="assemblyai_request_failed",
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AppError(
                f"AssemblyAI transcription request failed: {exc}",
                code="assemblyai_request_failed",
            ) from exc
        except json.JSONDecodeError as exc:
            raise AppError(
                "AssemblyAI transcription request returned an invalid response.",
                code="assemblyai_invalid_response",
            ) from exc

        # Poll for completion (with timeout)
        max_wait = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                poll_request = Request(
                    f"{transcript_url}/{transcript_id}",
                    headers=headers,
                    method="GET",
                )
                with urlopen(poll_request, timeout=30) as response:
                    poll_response = json.loads(response.read())
                    status = poll_response.get("status")
                    
                    if status == "completed":
                        words = poll_response.get("words", [])
                        if not words:
                            raise AppError(
                                "AssemblyAI returned an empty transcript.",
                                code="empty_transcript",
                            )
                        
                        print(f"  → Transcription complete: {len(words)} words")
                        
                        # Extract speaker information from words
                        has_speaker_labels = any(word.get("speaker") is not None for word in words)
                        unique_speakers = {_format_speaker_label(word.get('speaker')) for word in words if word.get('speaker') is not None}
                        
                        log_model_usage(
                            provider="AssemblyAI",
                            model="default",
                            purpose="speech-to-text transcription",
                            details=f"words={len(words)} has_speaker_labels={has_speaker_labels} speakers={len(unique_speakers)}",
                        )
                        
                        if unique_speakers:
                            print(f"  → Detected speakers: {unique_speakers}")
                        
                        return self._build_segments(
                            (
                                RawTranscriptSegment(
                                    start=float(word.get("start", 0) / 1000),  # Convert ms to seconds
                                    end=float(word.get("end", 0) / 1000),
                                    text=word.get("text", ""),
                                    speaker=_format_speaker_label(word.get('speaker')) if word.get('speaker') is not None else None,
                                )
                                for word in words
                            ),
                            default_speaker="Speaker 1",
                        )
                    elif status == "error":
                        error = poll_response.get("error", "Unknown error")
                        raise AppError(
                            f"AssemblyAI transcription error: {error}",
                            code="assemblyai_transcription_failed",
                        )
                    else:
                        pass
            except HTTPError as exc:
                raise AppError(
                    f"AssemblyAI poll failed: {_format_http_error(exc)}",
                    code="assemblyai_poll_failed",
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise AppError(
                    f"AssemblyAI poll failed: {exc}",
                    code="assemblyai_poll_failed",
                ) from exc
            except json.JSONDecodeError as exc:
                raise AppError(
                    "AssemblyAI poll returned an invalid response.",
                    code="assemblyai_invalid_response",
                ) from exc
            
            time.sleep(2)  # Wait 2 seconds before next poll
        
        raise AppError(
            "AssemblyAI transcription timed out.",
            code="assemblyai_transcription_timeout",
        )

    def _transcribe_with_faster_whisper(
        self,
        audio_path: Path,
    ) -> list[TranscriptSegment]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AppError(
                "Install faster-whisper to enable local transcription.",
                code="transcription_dependency_missing",
            ) from exc

        try:
            if self._model is None:
                self._model = WhisperModel(
                    self.settings.transcription_model,
                    device=self.settings.transcription_device,
                    compute_type=self.settings.transcription_compute_type,
                )

            segments, _info = self._model.transcribe(
                str(audio_path),
                language=self.settings.transcription_language,
                vad_filter=True,
                word_timestamps=False,
                beam_size=5,
            )
        except ValueError as exc:
            raise AppError(
                f"Invalid faster-whisper transcription configuration: {exc}",
                code="transcription_configuration_invalid",
            ) from exc
        except RuntimeError as exc:
            raise AppError(
                f"faster-whisper could not process the audio: {exc}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="transcription_runtime_error",
            ) from exc

        return self._build_segments(
            (
                RawTranscriptSegment(
                    start=float(segment.start or 0.0),
                    end=float(segment.end or 0.0),
                    text=str(segment.text or ""),
                )
                for segment in segments
            ),
            default_speaker=None,
        )

    def _transcribe_with_openai_whisper(
        self,
        audio_path: Path,
    ) -> list[TranscriptSegment]:
        try:
            import whisper
        except ImportError as exc:
            raise AppError(
                "Install openai-whisper to enable local transcription.",
                code="transcription_dependency_missing",
            ) from exc

        try:
            if self._model is None:
                self._model = whisper.load_model(
                    self.settings.transcription_model,
                    device=self.settings.transcription_device,
                )

            result = self._model.transcribe(
                str(audio_path),
                language=self.settings.transcription_language,
                fp16=self.settings.transcription_device != "cpu",
                verbose=False,
            )
        except ValueError as exc:
            raise AppError(
                f"Invalid whisper transcription configuration: {exc}",
                code="transcription_configuration_invalid",
            ) from exc
        except RuntimeError as exc:
            raise AppError(
                f"Whisper could not process the audio: {exc}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="transcription_runtime_error",
            ) from exc

        return self._build_segments(
            (
                RawTranscriptSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=str(segment.get("text", "")),
                )
                for segment in result.get("segments", [])
            ),
            default_speaker=None,
        )

    def _build_segments(
        self,
        raw_segments: Iterable["RawTranscriptSegment"],
        *,
        default_speaker: str | None,
    ) -> list[TranscriptSegment]:
        normalized_segments = [
            NormalizedTranscriptSegment(
                speaker=segment.speaker or default_speaker or "",
                start=max(0.0, segment.start),
                end=max(segment.start, segment.end),
                text=cleaned_text,
            )
            for segment in raw_segments
            if (cleaned_text := _clean_text(segment.text))
            and _is_useful_text(cleaned_text)
        ]
        if not normalized_segments:
            raise AppError(
                "No speech was detected in the uploaded audio.",
                code="empty_transcript",
            )

        # Merge consecutive segments from the same speaker at normalization stage
        merged_normalized = _merge_broken_segments(normalized_segments)

        # Convert merged normalized segments to final TranscriptSegment format with sequential IDs
        return [
            TranscriptSegment(
                segment_id=index,
                speaker=segment.speaker,
                start_seconds=round(segment.start, 2),
                end_seconds=round(segment.end, 2),
                start_time=_format_timestamp(segment.start),
                end_time=_format_timestamp(segment.end),
                text=_capitalize_first(segment.text),
            )
            for index, segment in enumerate(merged_normalized, start=1)
        ]


@dataclass(frozen=True)
class RawTranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None  # Optional speaker label from AssemblyAI or other sources


@dataclass(frozen=True)
class NormalizedTranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^\s*(\[[^\]]+\]|\([^)]+\))\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*(\[[^\]]+\]|\([^)]+\))\s*$", "", cleaned)
    cleaned = re.sub(r"<\|[^|]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" -_")
    if not cleaned:
        return ""
    cleaned = _fix_spacing_around_punctuation(cleaned)
    return cleaned


def _capitalize_first(text: str) -> str:
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _fix_spacing_around_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.!?;:%])", r"\1", text)
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()


def _is_useful_text(text: str) -> bool:
    lowered = text.lower().strip(" .!?")
    if len(lowered) < MIN_USEFUL_TEXT_CHARS:
        return False
    if lowered in COMMON_ASR_ARTIFACTS:
        return False
    return not any(re.match(pattern, lowered) for pattern in NOISE_ONLY_PATTERNS)


def _merge_broken_segments(
    segments: Iterable[NormalizedTranscriptSegment],
) -> list[NormalizedTranscriptSegment]:
    merged: list[NormalizedTranscriptSegment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        if _should_merge(previous, segment):
            merged[-1] = NormalizedTranscriptSegment(
                speaker=previous.speaker,
                start=previous.start,
                end=max(previous.end, segment.end),
                text=_join_segment_text(previous.text, segment.text),
            )
            continue

        merged.append(segment)

    return merged


def _should_merge(
    previous: NormalizedTranscriptSegment,
    current: NormalizedTranscriptSegment,
) -> bool:
    if previous.speaker != current.speaker:
        return False

    gap = max(0.0, current.start - previous.end)
    combined_length = len(previous.text) + len(current.text) + 1
    if gap > MAX_MERGE_GAP_SECONDS or combined_length > MAX_MERGED_SEGMENT_CHARS:
        return False

    current_text = current.text.lower()
    previous_is_open = previous.text[-1] not in ".?!"
    current_continues_sentence = current.text[:1].islower() or current_text.startswith(
        ("and ", "but ", "or ", "so ", "because ", "that ", "which ")
    )
    tiny_fragment = len(previous.text.split()) <= 4 or len(current.text.split()) <= 4

    return previous_is_open or current_continues_sentence or tiny_fragment


def _join_segment_text(previous: str, current: str) -> str:
    if not previous:
        return current
    if not current:
        return previous
    if previous.endswith("-"):
        return f"{previous[:-1]}{current}"
    return _fix_spacing_around_punctuation(f"{previous} {current}")


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_http_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    detail = re.sub(r"\s+", " ", body).strip()
    if detail:
        return f"HTTP {exc.code}: {detail[:300]}"
    return f"HTTP {exc.code}: {exc.reason or 'No error details returned.'}"


def _format_speaker_label(speaker: str | int) -> str:
    """Convert AssemblyAI speaker label to speaker name.
    
    Handles both numeric (0, 1, 2) and alphabetic (A, B, C) speaker labels.
    Returns Speaker 1, Speaker 2, Speaker 3, etc.
    """
    speaker_str = str(speaker).strip().upper()
    
    # Try to convert alphabetic label (A, B, C, etc.) to numeric
    if len(speaker_str) == 1 and speaker_str.isalpha():
        speaker_idx = ord(speaker_str) - ord('A')  # A=0, B=1, C=2, etc.
        return f"Speaker {speaker_idx + 1}"
    
    # Try to convert numeric string to int
    try:
        speaker_idx = int(speaker_str)
        return f"Speaker {speaker_idx + 1}"
    except ValueError:
        # Fallback if format is unexpected
        return f"Speaker {speaker_str}"

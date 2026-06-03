import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

from core.config import Settings
from core.errors import AppError
from core.usage_logging import log_model_usage
from schemas.audio import TranscriptSegment

MIN_OVERLAP_SECONDS = 0.05
TURN_SWITCH_GAP_SECONDS = 1.2
MIN_CONVERSATIONAL_TURN_GAP_SECONDS = 0.25
MAX_SAME_SPEAKER_MERGE_GAP_SECONDS = 1.25
MAX_MERGED_SEGMENT_CHARS = 900
CONTINUATION_PREFIXES = (
    "and ",
    "but ",
    "or ",
    "so ",
    "because ",
    "that ",
    "which ",
    "then ",
)


def _patch_huggingface_hub_download() -> None:
    """Patch huggingface_hub.hf_hub_download to handle deprecated use_auth_token parameter."""
    try:
        from huggingface_hub import hf_hub_download as original_hf_hub_download
        import huggingface_hub
    except ImportError:
        return

    if getattr(original_hf_hub_download, "_voice_intelligence_patch", False):
        return

    def compatible_hf_hub_download(*args: object, **kwargs: object) -> object:
        # Convert deprecated use_auth_token to token parameter
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original_hf_hub_download(*args, **kwargs)

    compatible_hf_hub_download._voice_intelligence_patch = True
    huggingface_hub.hf_hub_download = compatible_hf_hub_download


# Apply the patch at module import time
_patch_huggingface_hub_download()


class SpeakerDiarizationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline = None

    async def diarize(
        self,
        transcript: list[TranscriptSegment],
        audio_path: Path | None = None,
    ) -> list[TranscriptSegment]:
        if not transcript:
            return []

        # Check for existing speaker labels from Assembly AI transcription
        unique_speakers = {segment.speaker for segment in transcript if segment.speaker.strip()}
        has_any_speakers = len(unique_speakers) > 0
        
        print(f"\n📊 [DIARIZATION] Found {len(unique_speakers)} unique speaker(s): {unique_speakers}")
        
        # If Assembly AI detected any speakers (1 or more), use them
        if has_any_speakers:
            print("✓ [DIARIZATION] Using speaker labels from Assembly AI transcription")
            log_model_usage(
                provider="transcript",
                model="existing-speaker-labels",
                purpose="speaker diarization",
                details=f"segments={len(transcript)} speakers={len(unique_speakers)}",
            )
            print("✅ [DIARIZATION] Reused existing speaker labels from AssemblyAI successfully.")
            return _renumber_segments(_merge_consecutive_speaker_segments(transcript))

        # Only try pyannote if Assembly AI found NO speakers (API failure or no speech)
        engine = self.settings.diarization_engine.strip().lower()
        if engine == "pyannote" and audio_path is not None:
            print(f"🔊 [DIARIZATION] Attempting Pyannote diarization ({self.settings.diarization_model})...")
            log_model_usage(
                provider="Hugging Face",
                model=self.settings.diarization_model,
                purpose="speaker diarization",
                details=f"segments={len(transcript)} (fallback from Assembly AI failure)",
            )
            diarized = await self._try_pyannote_diarization(transcript, audio_path)
            if diarized:
                print(f"✅ [DIARIZATION] Pyannote speaker diarization model ({self.settings.diarization_model}) succeeded.")
                return diarized
            # Fall back to heuristic if pyannote also fails
            print(f"❌ [DIARIZATION] Pyannote speaker diarization model ({self.settings.diarization_model}) failed. 📍 Fallback: using local heuristic turn-switching diarization.")
            log_model_usage(
                provider="local",
                model="heuristic-turn-switching",
                purpose="speaker diarization",
                mode="fallback",
                details="both assembly ai and pyannote unavailable or returned no speaker turns",
            )
            return self._heuristic_diarization(transcript)

        print("📍 [DIARIZATION] Pyannote speaker diarization not configured or unavailable. Fallback: using local heuristic turn-switching diarization.")
        log_model_usage(
            provider="local",
            model="heuristic-turn-switching",
            purpose="speaker diarization",
            details=f"segments={len(transcript)} (fallback from Assembly AI failure)",
        )
        return self._heuristic_diarization(transcript)

    async def _try_assemblyai_diarization(
        self,
        transcript: list[TranscriptSegment],
        audio_path: Path,
    ) -> list[TranscriptSegment] | None:
        try:
            return await asyncio.to_thread(
                self._diarize_with_assemblyai,
                transcript,
                audio_path,
            )
        except Exception as exc:
            # Log the error but return None to fall back to other methods
            print(f"    Error during AssemblyAI diarization: {exc}")
            return None

    def _diarize_with_assemblyai(
        self,
        transcript: list[TranscriptSegment],
        audio_path: Path,
    ) -> list[TranscriptSegment] | None:
        """Diarize audio using AssemblyAI cloud API."""
        import json
        import time
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

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
            print(f"    → Uploading audio file ({len(audio_data) / 1024 / 1024:.1f}MB)...")
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
                    raise AppError(f"AssemblyAI upload failed: no upload_url in response")
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            raise AppError(f"AssemblyAI upload failed: {exc}") from exc

        # Request diarization with speaker labels
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        transcript_data = {
            "audio_url": audio_url,
            "speaker_labels": True,  # Enable speaker diarization
        }
        
        try:
            print(f"    → Requesting diarization...")
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
                    raise AppError(f"AssemblyAI request failed: no ID in response")
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            raise AppError(f"AssemblyAI request failed: {exc}") from exc

        # Poll for completion
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
                            print("    Warning: AssemblyAI returned empty transcript")
                            return None
                        
                        # Extract speaker information from words
                        has_speakers = any(word.get("speaker") is not None for word in words)
                        if not has_speakers:
                            print("    Warning: AssemblyAI did not detect multiple speakers")
                            return None
                        
                        print(f"    → Diarization complete: {len(words)} words detected")
                        
                        log_model_usage(
                            provider="AssemblyAI",
                            model="default",
                            purpose="speaker diarization",
                            details=f"words={len(words)} has_speaker_labels={has_speakers}",
                        )
                        
                        # Build diarization from speaker labels
                        turns = _normalize_assemblyai_speakers(words)
                        if not turns:
                            print("    Warning: Could not extract speaker turns from AssemblyAI")
                            return None
                        
                        speaker_names = _speaker_name_map(turns)
                        diarized_segments = [
                            _copy_segment(
                                segment,
                                speaker=_best_speaker_for_segment(segment, turns, speaker_names),
                            )
                            for segment in transcript
                        ]
                        return _renumber_segments(_merge_consecutive_speaker_segments(diarized_segments))
                        
                    elif status == "error":
                        error = poll_response.get("error", "Unknown error")
                        raise AppError(f"AssemblyAI diarization error: {error}")
                    else:
                        print(f"    → Status: {status}... waiting...")
            except (HTTPError, URLError, json.JSONDecodeError) as exc:
                raise AppError(f"AssemblyAI poll failed: {exc}") from exc
            
            time.sleep(2)  # Wait 2 seconds before next poll
        
        raise AppError("AssemblyAI diarization timeout")

    async def _try_pyannote_diarization(
        self,
        transcript: list[TranscriptSegment],
        audio_path: Path,
    ) -> list[TranscriptSegment] | None:
        if not self.settings.diarization_auth_token:
            return None

        try:
            return await asyncio.to_thread(
                self._diarize_with_pyannote,
                transcript,
                audio_path,
            )
        except AppError as exc:
            # Re-raise AppError only if it's an authentication issue
            if "auth" in exc.code.lower() or "token" in exc.message.lower():
                raise
            # Otherwise, return None to fall back to heuristic diarization
            logger.warning(f"Pyannote diarization failed: {exc}")
            return None
        except Exception as exc:
            # Log the error but return None to fall back to heuristic
            logger.exception("Pyannote diarization failed with unexpected error:")
            return None

    def _diarize_with_pyannote(
        self,
        transcript: list[TranscriptSegment],
        audio_path: Path,
    ) -> list[TranscriptSegment] | None:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise AppError(
                "Install pyannote.audio from requirements-diarization.txt to enable speaker diarization.",
                code="diarization_dependency_missing",
            ) from exc

        if self._pipeline is None:
            if not self.settings.diarization_auth_token:
                raise AppError(
                    "Pyannote diarization requires HUGGINGFACE_TOKEN, HF_TOKEN, or PYANNOTE_AUTH_TOKEN to be set.",
                    code="diarization_auth_missing",
                )
            
            try:
                # Login with HuggingFace token for gated models
                from huggingface_hub import login
                login(token=self.settings.diarization_auth_token, add_to_git_credential=False)
                
                # Now load the model (will use the authenticated session)
                self._pipeline = Pipeline.from_pretrained(self.settings.diarization_model)
            except Exception as exc:
                raise AppError(
                    f"Failed to load pyannote model '{self.settings.diarization_model}': {exc}. "
                    "Check that your HuggingFace token is valid and has been granted access to the gated model.",
                    code="diarization_model_load_failed",
                ) from exc

        diarization = self._pipeline(
            str(audio_path),
            min_speakers=self.settings.diarization_min_speakers,
            max_speakers=self.settings.diarization_max_speakers,
        )
        turns = _normalize_pyannote_turns(diarization)
        if not turns:
            return None

        speaker_names = _speaker_name_map(turns)
        diarized_segments = [
            _copy_segment(
                segment,
                speaker=_best_speaker_for_segment(segment, turns, speaker_names),
            )
            for segment in transcript
        ]
        return _renumber_segments(_merge_consecutive_speaker_segments(diarized_segments))

    def _heuristic_diarization(
        self,
        transcript: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        speaker_count = max(1, self.settings.diarization_default_speakers)
        speaker_index = 0
        diarized: list[TranscriptSegment] = []

        for index, segment in enumerate(transcript):
            if index > 0 and _looks_like_speaker_turn(transcript[index - 1], segment):
                speaker_index = (speaker_index + 1) % speaker_count

            diarized.append(
                _copy_segment(segment, speaker=f"Speaker {speaker_index + 1}")
            )

        return _renumber_segments(_merge_consecutive_speaker_segments(diarized))


@dataclass(frozen=True)
class DiarizationTurn:
    start: float
    end: float
    source_speaker: str


def _normalize_assemblyai_speakers(words: list[dict]) -> list[DiarizationTurn]:
    """Convert AssemblyAI word-level speaker data into speaker turns.
    
    Handles both numeric (0, 1, 2) and alphabetic (A, B, C) speaker labels.
    Uses speaker indices directly for mapping.
    """
    turns: list[DiarizationTurn] = []
    current_speaker_idx: str | None = None
    turn_start: float | None = None

    for word in words:
        speaker = word.get("speaker")
        if speaker is None:
            continue
        
        # Convert alphabetic label (A, B, C) to numeric (0, 1, 2) for consistency
        speaker_idx = _convert_speaker_label_to_index(speaker)
        
        start_ms = word.get("start", 0)
        end_ms = word.get("end", 0)
        start_seconds = float(start_ms / 1000) if start_ms else 0.0
        end_seconds = float(end_ms / 1000) if end_ms else start_seconds
        
        # If speaker changed, save the previous turn
        if speaker_idx != current_speaker_idx:
            if current_speaker_idx is not None and turn_start is not None:
                # Find the end time of the last word with current_speaker_idx
                last_end = turn_start
                for w in words:
                    w_speaker = w.get("speaker")
                    if w_speaker is not None:
                        w_speaker_idx = _convert_speaker_label_to_index(w_speaker)
                        if w_speaker_idx == current_speaker_idx:
                            w_end_ms = w.get("end", 0)
                            w_end_seconds = float(w_end_ms / 1000) if w_end_ms else 0.0
                            last_end = w_end_seconds
                
                turns.append(
                    DiarizationTurn(
                        start=turn_start,
                        end=last_end,
                        source_speaker=str(current_speaker_idx),
                    )
                )
            
            current_speaker_idx = speaker_idx
            turn_start = start_seconds

    # Add the final turn
    if current_speaker_idx is not None and turn_start is not None:
        last_end = turn_start
        for w in words:
            w_speaker = w.get("speaker")
            if w_speaker is not None:
                w_speaker_idx = _convert_speaker_label_to_index(w_speaker)
                if w_speaker_idx == current_speaker_idx:
                    w_end_ms = w.get("end", 0)
                    w_end_seconds = float(w_end_ms / 1000) if w_end_ms else 0.0
                    last_end = w_end_seconds
        
        turns.append(
            DiarizationTurn(
                start=turn_start,
                end=last_end,
                source_speaker=str(current_speaker_idx),
            )
        )
    
    return sorted(turns, key=lambda item: (item.start, item.end))


def _normalize_pyannote_turns(diarization: object) -> list[DiarizationTurn]:
    turns: list[DiarizationTurn] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            DiarizationTurn(
                start=float(turn.start),
                end=float(turn.end),
                source_speaker=str(speaker),
            )
        )
    return sorted(turns, key=lambda item: (item.start, item.end))


def _speaker_name_map(turns: list[DiarizationTurn]) -> dict[str, str]:
    """Map speaker IDs to speaker names, starting from Speaker 1.
    
    Always uses sequential naming based on appearance order: Speaker 1, Speaker 2, Speaker 3, etc.
    Works for both AssemblyAI (numeric) and pyannote (non-numeric) speaker IDs.
    """
    names: dict[str, str] = {}
    for turn in turns:
        if turn.source_speaker not in names:
            # Assign next speaker number based on order of appearance
            names[turn.source_speaker] = f"Speaker {len(names) + 1}"
    return names


def _best_speaker_for_segment(
    segment: TranscriptSegment,
    turns: list[DiarizationTurn],
    speaker_names: dict[str, str],
) -> str:
    overlap_by_speaker: dict[str, float] = {}
    for turn in turns:
        overlap = _overlap_seconds(
            segment.start_seconds,
            segment.end_seconds,
            turn.start,
            turn.end,
        )
        if overlap < MIN_OVERLAP_SECONDS:
            continue
        overlap_by_speaker[turn.source_speaker] = (
            overlap_by_speaker.get(turn.source_speaker, 0.0) + overlap
        )

    if overlap_by_speaker:
        source_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        return speaker_names[source_speaker]

    nearest_turn = min(
        turns,
        key=lambda turn: min(
            abs(segment.start_seconds - turn.end),
            abs(segment.end_seconds - turn.start),
        ),
    )
    return speaker_names[nearest_turn.source_speaker]


def _overlap_seconds(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _looks_like_speaker_turn(
    previous: TranscriptSegment,
    current: TranscriptSegment,
) -> bool:
    gap = max(0.0, current.start_seconds - previous.end_seconds)
    if gap >= TURN_SWITCH_GAP_SECONDS:
        return True
    if gap < MIN_CONVERSATIONAL_TURN_GAP_SECONDS:
        return False

    previous_text = previous.text.strip()
    current_text = current.text.strip()
    if not previous_text or not current_text:
        return False

    current_lower = current_text.lower()
    if current_text[:1].islower() or current_lower.startswith(CONTINUATION_PREFIXES):
        return False

    if previous_text.endswith("?"):
        return True

    return previous_text.endswith((".", "!", ":"))


def _has_meaningful_speaker_labels(transcript: list[TranscriptSegment]) -> bool:
    """Check if transcript has meaningful speaker labels (more than one unique speaker)."""
    speakers = {segment.speaker for segment in transcript if segment.speaker.strip()}
    return len(speakers) > 1


def _copy_segment(segment: TranscriptSegment, speaker: str) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment.segment_id,
        speaker=speaker,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        start_time=segment.start_time,
        end_time=segment.end_time,
        text=segment.text,
    )


def _merge_consecutive_speaker_segments(
    transcript: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in transcript:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        gap = max(0.0, segment.start_seconds - previous.end_seconds)
        combined_length = len(previous.text) + len(segment.text) + 1
        if (
            previous.speaker == segment.speaker
            and gap <= MAX_SAME_SPEAKER_MERGE_GAP_SECONDS
            and combined_length <= MAX_MERGED_SEGMENT_CHARS
        ):
            merged[-1] = TranscriptSegment(
                segment_id=previous.segment_id,
                speaker=previous.speaker,
                start_seconds=previous.start_seconds,
                end_seconds=segment.end_seconds,
                start_time=previous.start_time,
                end_time=segment.end_time,
                text=_join_text(previous.text, segment.text),
            )
            continue

        merged.append(segment)

    return merged


def _join_text(previous: str, current: str) -> str:
    if not previous:
        return current
    if not current:
        return previous
    return f"{previous.rstrip()} {current.lstrip()}"


def _renumber_segments(
    transcript: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    # Renumber segment IDs sequentially
    # Also renumber speakers to be sequential (Speaker 1, 2, 3...) based on first appearance
    speaker_mapping: dict[str, str] = {}
    renumbered: list[TranscriptSegment] = []
    
    for index, segment in enumerate(transcript, start=1):
        # Map speakers to sequential numbering if needed
        speaker = segment.speaker
        if speaker not in speaker_mapping:
            # Assign next speaker number based on how many unique speakers we've seen
            next_speaker_num = len(speaker_mapping) + 1
            speaker_mapping[speaker] = f"Speaker {next_speaker_num}"
        
        renumbered_speaker = speaker_mapping[speaker]
        
        renumbered.append(
            TranscriptSegment(
                segment_id=index,
                speaker=renumbered_speaker,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
            )
        )
    return renumbered


def _convert_speaker_label_to_index(speaker: str | int) -> str:
    """Convert AssemblyAI speaker label to numeric index string.
    
    Handles both numeric (0, 1, 2) and alphabetic (A, B, C) labels.
    Returns numeric index as string for consistent comparison.
    """
    speaker_str = str(speaker).strip().upper()
    
    # If already numeric, return as-is
    try:
        int(speaker_str)
        return speaker_str
    except ValueError:
        pass
    
    # Convert alphabetic label (A, B, C, etc.) to numeric (0, 1, 2, etc.)
    if len(speaker_str) == 1 and speaker_str.isalpha():
        speaker_idx = ord(speaker_str) - ord('A')
        return str(speaker_idx)
    
    # Fallback: return as-is
    return speaker_str

import asyncio
from pathlib import Path
import shutil
import subprocess
import tempfile

from starlette import status

from core.config import Settings
from core.errors import AppError


class AudioPreprocessor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def prepare_for_ai(self, input_path: Path, audio_id: str) -> Path:
        output_path = _temporary_processed_audio_path(audio_id)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AppError(
                "Could not create a temporary processed audio directory.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="temporary_processed_audio_directory_unavailable",
            ) from exc

        if not input_path.exists():
            raise AppError(
                "Uploaded audio file was not found.",
                code="audio_not_found",
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            if input_path.suffix.lower() == ".wav":
                try:
                    await asyncio.to_thread(shutil.copyfile, input_path, output_path)
                except OSError as exc:
                    raise AppError(
                        "Audio preprocessing failed while copying the WAV file.",
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        code="audio_preprocessing_failed",
                    ) from exc
                return output_path

            raise AppError(
                "FFmpeg is required to preprocess mp3 and m4a audio files.",
                code="ffmpeg_not_available",
            )

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(self.settings.transcription_channels),
            "-ar",
            str(self.settings.transcription_sample_rate),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(output_path),
        ]

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise AppError(
                "Audio preprocessing could not start FFmpeg.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="audio_preprocessing_unavailable",
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            details = f" Details: {stderr[:300]}" if stderr else ""
            raise AppError(
                f"Audio preprocessing failed.{details}",
                code="audio_preprocessing_failed",
            )

        return output_path


def _temporary_processed_audio_path(audio_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "voice_intelligence" / "processed" / f"{audio_id}.wav"

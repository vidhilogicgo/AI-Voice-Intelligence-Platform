from pathlib import Path
import tempfile
import time

from fastapi import UploadFile
from starlette import status

from core.config import Settings
from core.errors import AppError


class AudioHandler:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def save_upload(self, file: UploadFile, file_id: str) -> Path:
        target_path: Path | None = None
        try:
            extension = Path(file.filename or "").suffix.lower()
            if not file.filename or not extension:
                raise AppError(
                    "Uploaded audio file must include a valid filename and extension.",
                    code="missing_audio_filename",
                )

            if extension not in self.settings.allowed_audio_extensions:
                allowed = ", ".join(sorted(self.settings.allowed_audio_extensions))
                raise AppError(
                    f"Unsupported audio format. Allowed formats: {allowed}.",
                    code="unsupported_audio_format",
                )

            target_path = _temporary_audio_path(file_id, extension, "uploads")
            max_bytes = self.settings.max_upload_mb * 1024 * 1024
            total_bytes = 0
            with target_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise AppError(
                            f"Audio file is larger than {self.settings.max_upload_mb} MB.",
                            code="file_too_large",
                        )
                    output.write(chunk)
        except AppError:
            # Clean up file on AppError (like file_too_large)
            if target_path is not None:
                self._safe_delete_file(target_path)
            raise
        except Exception as exc:
            # Clean up on any other error
            if target_path is not None:
                self._safe_delete_file(target_path)
            raise AppError(
                "Audio upload could not be saved. Please try again.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="audio_upload_save_failed",
            ) from exc
        finally:
            try:
                await file.close()
            except Exception:
                pass

        return target_path

    def _safe_delete_file(self, file_path: Path) -> None:
        """Safely delete a file with retry logic for Windows permission issues."""
        if not file_path.exists():
            return
        
        # Try to delete with retries for Windows file locking issues
        for attempt in range(3):
            try:
                file_path.unlink(missing_ok=True)
                return
            except PermissionError as exc:
                if attempt < 2:
                    # Wait a bit and retry
                    time.sleep(0.1)
                else:
                    # Log but don't raise - file will be cleaned up eventually
                    print(f"[AUDIO_HANDLER] Warning: Could not delete file {file_path}: {exc}")
            except Exception as exc:
                print(f"[AUDIO_HANDLER] Warning: Error deleting file {file_path}: {exc}")
                return


def _temporary_audio_path(file_id: str, extension: str, category: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "voice_intelligence" / category
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppError(
            "Could not create a temporary audio directory.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="temporary_audio_directory_unavailable",
        ) from exc
    return temp_dir / f"{file_id}{extension}"

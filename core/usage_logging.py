from datetime import datetime
import re
from typing import Any

_mongo_logs_collection: Any | None = None
_mongo_logs_checked = False


def log_model_usage(
    *,
    provider: str,
    model: str,
    purpose: str,
    mode: str = "model",
    details: str | None = None,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"[{timestamp}] [model-usage] provider={provider} "
        f"mode={mode} model={model} purpose={purpose}"
    )
    if details:
        message = f"{message} details={details}"
    print(message, flush=True)
    _persist_model_usage(
        provider=provider,
        model=model,
        purpose=purpose,
        mode=mode,
        details=details,
        timestamp=timestamp,
    )


def log_model_fallback(
    *,
    provider: str,
    model: str,
    purpose: str,
    reason: str,
) -> None:
    log_model_usage(
        provider=provider,
        model=model,
        purpose=purpose,
        mode="fallback",
        details=f"reason={reason}",
    )


def _persist_model_usage(
    *,
    provider: str,
    model: str,
    purpose: str,
    mode: str,
    details: str | None,
    timestamp: str,
) -> None:
    collection = _get_mongo_logs_collection()
    if collection is None:
        return

    try:
        collection.insert_one(
            {
                "audio_id": _extract_audio_id(details),
                "provider": provider,
                "model": model,
                "purpose": purpose,
                "mode": mode,
                "details": details,
                "created_at": timestamp,
            }
        )
    except Exception as exc:
        print(f"[model-usage] MongoDB log write failed: {exc}", flush=True)


def _get_mongo_logs_collection() -> Any | None:
    global _mongo_logs_checked, _mongo_logs_collection
    if _mongo_logs_checked:
        return _mongo_logs_collection

    _mongo_logs_checked = True
    try:
        from core.config import get_settings
        from pymongo import MongoClient

        settings = get_settings()
        if not settings.mongodb_uri:
            return None

        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        _mongo_logs_collection = client[settings.mongodb_db_name].model_usage_logs
        _mongo_logs_collection.create_index("audio_id")
        return _mongo_logs_collection
    except Exception as exc:
        print(f"[model-usage] MongoDB logging disabled: {exc}", flush=True)
        return None


def _extract_audio_id(details: str | None) -> str | None:
    if not details:
        return None
    match = re.search(r"\baudio_id=([a-fA-F0-9-]{36})\b", details)
    if match is None:
        return None
    return match.group(1)

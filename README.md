# AI Voice Intelligence API

FastAPI backend for uploading recorded calls, transcribing audio, separating speakers, generating summaries and business insights, and answering transcript-based questions.

## Features

- Audio upload for `.mp3`, `.wav`, and `.m4a`
- Temporary local audio handling with cleanup after processing
- Speech-to-text transcription with AssemblyAI fallback/local Whisper support
- Speaker diarization
- Summary and insight extraction
- Transcript-based Q&A
- MongoDB persistence for audio status, results, processing logs, Q&A history, model usage logs, and live status

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Fill in required values in `.env`, especially:

```env
MONGODB_URI=...
MONGODB_DB_NAME=ai_voice_intelligence
GROQ_API_KEY=...
```

## Run

Local-only:

```powershell
uvicorn app:app --port 8000
```

Accessible from another device on the same network:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```text
GET /api/health
```

## Main Endpoints

```text
POST /api/audio/upload
GET  /api/audio/{audio_id}/status
GET  /api/audio/{audio_id}/result
POST /api/audio/{audio_id}/ask
GET  /api/live-status
```

## MongoDB Collections

Currently used collections:

```text
audio_files
results
processing_logs
qa_history
model_usage_logs
live_status
```

Runtime audio files are not stored permanently in the repo. Uploaded and processed files are written to the system temp directory and deleted after processing.

## GitHub Safety

Do not commit `.env`, `.venv`, `storage`, cache folders, or uploaded audio. They are ignored by `.gitignore`.

If a real API key or MongoDB URI is ever pasted into chat or committed by mistake, rotate the credential immediately.

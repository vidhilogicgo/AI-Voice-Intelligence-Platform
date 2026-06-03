# AI Voice Intelligence API

FastAPI backend for an AI sales and research call intelligence MVP. The API accepts recorded calls, preprocesses audio, transcribes speech, labels speakers, generates summaries and business insights, supports transcript Q&A, and stores processing data in MongoDB.

## MVP Capabilities

- Upload `.mp3`, `.wav`, or `.m4a` call recordings.
- Track processing status and current processing state.
- Convert audio into timestamped transcript segments.
- Identify speakers with either transcript speaker labels, heuristic fallback, or optional pyannote diarization.
- Generate a short summary, detailed summary, and key discussion points.
- Extract business insights such as pain points, objections, requirements, feature requests, sentiment, buying intent, and action items.
- Ask questions against the completed transcript.
- Store audio metadata, combined results, processing logs, Q&A history, model usage logs, and live status metrics in MongoDB.
- Clean up temporary uploaded and processed audio files after processing.

## Project Setup

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

Fill `.env` with your local values. Do not commit `.env`.

Run locally:

```powershell
uvicorn app:app --port 8000
```

Run on your LAN so another device can call the API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then use:

```text
http://<your-ip-address>:8000/api/health
```

## Required Dependencies

Dependencies are listed in `requirements.txt`:

- `fastapi`: API framework.
- `uvicorn[standard]`: ASGI server.
- `python-multipart`: file upload support.
- `faster-whisper`: local open-source speech-to-text engine.
- `numpy` and `protobuf`: model/runtime support.
- `pymongo`: MongoDB persistence.
- `certifi`: CA certificate bundle for MongoDB Atlas TLS.
- `faiss-cpu`: local vector search for transcript Q&A.
- `sentence-transformers`: local embedding model support for semantic retrieval.
- `torch` and `torchaudio`: model runtime for pyannote.
- `pyannote.audio`: neural speaker diarization.

Pyannote is only used when a Hugging Face token is configured. Without that token, the app falls back to heuristic diarization.

FFmpeg is recommended for audio conversion and normalization. If FFmpeg is not installed, `.wav` files can still be copied through preprocessing, but `.mp3` and `.m4a` preprocessing requires FFmpeg.

### Environment Variables

The application reads only a specific set of credentials and connection URIs from the environment. All other application settings use hardcoded default values.

| Variable | Required | Purpose |
| --- | --- | --- |
| `MONGODB_URI` | Recommended | MongoDB connection URI. Falls back to in-memory store if unavailable. |
| `MONGODB_DB_NAME` | Recommended | Database name, default `ai_voice_intelligence`. |
| `GROQ_API_KEY` | Recommended | Groq API key for summary, insight, and Q&A generation. |
| `ASSEMBLYAI_API_KEY` | Optional | Approved speech-to-text API key. If present, cloud transcription and diarization are used. |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | Required for pyannote | Token for Hugging Face to access the gated pyannote speaker diarization models. |

Example `.env` file:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
MONGODB_DB_NAME=ai_voice_intelligence
GROQ_API_KEY=gsk_...
ASSEMBLYAI_API_KEY=b1a...
HF_TOKEN=hf_...
```

## Models, Tools, and APIs Used

| Component | Model/Tool/API | Why it is used |
| --- | --- | --- |
| Audio preprocessing | FFmpeg | Reliable conversion to normalized mono 16 kHz WAV for downstream speech models. |
| Local transcription | `faster-whisper` with model such as `base` | Open-source, efficient local speech-to-text for MVP use without requiring a paid transcription API. |
| Optional transcription API | AssemblyAI | Approved external API option that can return high-quality transcripts and speaker labels. |
| Speaker diarization | Existing transcript speaker labels | If AssemblyAI or another transcript source already provides labels, reuse them instead of rerunning diarization. |
| Speaker diarization fallback | Heuristic diarization | Keeps the MVP functional when no pyannote/Hugging Face setup is available. |
| Optional neural diarization | `pyannote/speaker-diarization-3.1` | More advanced open-source speaker diarization when a Hugging Face token and dependencies are configured. |
| Summary generation | Groq chat model, default `llama-3.1-8b-instant` | Fast approved API for business-readable summaries. |
| Insight extraction | Groq chat model, default `llama-3.1-8b-instant` | Prompt-based extraction of business fields from raw call text. |
| Q&A retrieval | FAISS with `sentence-transformers/all-MiniLM-L6-v2` | Semantic transcript search that can match meaning, not only exact words. |
| Q&A fallback retrieval | Local TF-IDF sparse vector retrieval | Keeps Q&A usable if FAISS or the embedding model is unavailable on a demo machine. |
| Q&A answer generation | Groq chat model | Converts retrieved transcript context into a readable answer. |
| Persistence | MongoDB | Stores job metadata, combined results, processing logs, Q&A history, model usage, and live status. |

## MongoDB Collections

The MVP uses these collections:

```text
audio_files
results
processing_logs
qa_history
model_usage_logs
live_status
```

`results` stores transcript, summary, and insights together under the same `audio_id`.

Example `results` document shape:

```json
{
  "_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
  "audio_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
  "transcript": {
    "full_text": "Customer says onboarding is slow...",
    "speakers": ["Speaker 1", "Speaker 2"],
    "speakers_count": 2,
    "segment_count": 2,
    "segments": [
      {
        "segment_id": 1,
        "speaker": "Speaker 1",
        "start_seconds": 0.0,
        "end_seconds": 8.4,
        "start_time": "00:00",
        "end_time": "00:08",
        "text": "We are struggling with onboarding our sales team quickly."
      }
    ]
  },
  "summary": {
    "short": "The customer is evaluating a sales enablement solution because onboarding is slow and reporting is inconsistent.",
    "detailed": "The call focused on onboarding delays, CRM visibility gaps, and the need for better sales manager reporting...",
    "key_points": [
      "Customer wants faster onboarding for new sales representatives.",
      "Managers need clearer call and pipeline visibility.",
      "Follow-up demo should focus on reporting and onboarding workflows."
    ]
  },
  "insights": {
    "pain_points": ["Slow onboarding", "Limited manager visibility"],
    "objections": ["Concerned about implementation effort"],
    "requirements": ["CRM integration", "Manager dashboard"],
    "feature_requests": ["Automated call summaries"],
    "sentiment": "Interested but cautious",
    "buying_intent": "Medium",
    "action_items": ["Schedule reporting-focused demo"]
  },
  "created_at": "2026-06-02T10:00:00Z",
  "updated_at": "2026-06-02T10:01:30Z"
}
```

## API List

### Health Check

```text
GET /api/health
```

Sample response:

```json
{
  "success": true,
  "message": "Service is healthy.",
  "data": {
    "status": "ok",
    "service": "ai-voice-intelligence-api"
  }
}
```

### Upload Audio

```text
POST /api/audio/upload
Content-Type: multipart/form-data
```

Form field:

```text
file=<audio file>
```

Sample `curl`:

```bash
curl -X POST "http://localhost:8000/api/audio/upload" \
  -F "file=@call_recording.wav"
```

Sample response:

```json
{
  "success": true,
  "message": "Audio upload accepted and pending processing.",
  "data": {
    "audio_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
    "filename": "call_recording.wav",
    "status": "pending"
  }
}
```

Unsupported file sample response:

```json
{
  "success": false,
  "message": "Unsupported audio format. Allowed formats: .m4a, .mp3, .wav."
}
```

### Check Processing Status

```text
GET /api/audio/{audio_id}/status
```

Sample response while processing:

```json
{
  "success": true,
  "message": "Processing status fetched successfully.",
  "data": {
    "audio_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
    "status": "processing",
    "state": "transcription"
  }
}
```

Possible `status` values:

```text
pending
processing
completed
failed
```

Common `state` values:

```text
processing_started
preprocessing
transcription
diarization
summarization
insight_extraction
completed
preprocessing_failed
transcription_failed
diarization_failed
summarization_failed
insight_extraction_failed
```

### Get Completed Result

```text
GET /api/audio/{audio_id}/result
```

Sample response:

```json
{
  "success": true,
  "message": "Analysis result fetched successfully.",
  "data": {
    "audio_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
    "status": "completed",
    "result": {
      "transcript": [
        {
          "segment_id": 1,
          "speaker": "Speaker 1",
          "start_seconds": 0.0,
          "end_seconds": 8.4,
          "start_time": "00:00",
          "end_time": "00:08",
          "text": "We are struggling with onboarding our sales team quickly."
        }
      ],
      "summary": {
        "short": "The customer is evaluating a sales enablement solution because onboarding is slow and reporting is inconsistent.",
        "detailed": "The call focused on onboarding delays, CRM visibility gaps, and better manager reporting...",
        "key_points": [
          "Customer wants faster onboarding.",
          "Reporting visibility is a major requirement.",
          "A follow-up demo should focus on manager workflows."
        ]
      },
      "insights": {
        "pain_points": ["Slow onboarding", "Limited reporting visibility"],
        "objections": ["Concerned about implementation effort"],
        "requirements": ["CRM integration", "Manager dashboard"],
        "feature_requests": ["Automated call summaries"],
        "sentiment": "Interested but cautious",
        "buying_intent": "Medium",
        "action_items": ["Schedule reporting-focused demo"]
      }
    }
  }
}
```

If the result is not ready:

```json
{
  "success": false,
  "message": "Analysis result is not ready yet."
}
```

### Ask a Transcript Question

```text
POST /api/audio/{audio_id}/ask
Content-Type: application/json
```

Sample request:

```json
{
  "question": "What objections did the customer raise?"
}
```

Sample response:

```json
{
  "success": true,
  "message": "Question answered successfully.",
  "data": {
    "audio_id": "9c6a0d3d-32f2-4a23-bd84-2a1ad2b06d12",
    "question": "What objections did the customer raise?",
    "answer": "The customer was mainly concerned about implementation effort and whether the CRM integration would require extra manual work.",
    "sources": [
      {
        "segment_id": 4,
        "speaker": "Speaker 2",
        "start_seconds": 31.2,
        "end_seconds": 42.7,
        "start_time": "00:31",
        "end_time": "00:43",
        "text": "My concern is how much setup work this will take and whether it connects cleanly with our CRM."
      }
    ]
  }
}
```

### Live Status Metrics

```text
GET /api/live-status
```

Sample response:

```json
{
  "success": true,
  "message": "Live status fetched successfully.",
  "data": {
    "success_calls": 12,
    "failed_calls": 2,
    "total_calls": 14,
    "success_rate": 85.71,
    "failure_rate": 14.29,
    "avg_response_time_ms": 18342.25
  }
}
```

## Error Handling Behavior

- Failed upload and unsupported file errors return safe API messages.
- Preprocessing, transcription, diarization, summarization, and insight extraction failures mark the job as `failed`.
- Status responses expose `status` and `state` so the frontend can show progress.
- Processing logs store debugging context in MongoDB without exposing raw stack traces in API responses.
- Failed API responses do not include a `data` object.

## MVP Walkthrough / Demo

1. Start the API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. Verify the service:

```text
GET http://localhost:8000/api/health
```

3. Upload a call recording:

```bash
curl -X POST "http://localhost:8000/api/audio/upload" \
  -F "file=@call_recording.wav"
```

4. Copy the returned `audio_id`.

5. Poll processing status:

```text
GET http://localhost:8000/api/audio/{audio_id}/status
```

The `state` field moves through preprocessing, transcription, diarization, summarization, and insight extraction.

6. Fetch the completed result:

```text
GET http://localhost:8000/api/audio/{audio_id}/result
```

The response contains transcript segments, summary, and business insights in one JSON payload.

7. Ask a follow-up question:

```text
POST http://localhost:8000/api/audio/{audio_id}/ask
```

Body:

```json
{
  "question": "What follow-up actions should the sales team take?"
}
```

8. Check operational metrics:

```text
GET http://localhost:8000/api/live-status
```

## Known Limitations

- FastAPI `BackgroundTasks` are suitable for the MVP, but production should use a durable queue such as Celery, RQ, Dramatiq, or a cloud queue.
- If MongoDB is unavailable, the app falls back to in-memory storage, which is not durable.
- Local transcription speed depends on CPU/GPU resources and selected model size.
- Heuristic diarization is less accurate than pyannote or transcript speaker labels.
- Pyannote requires additional dependencies and access to Hugging Face gated models.
- Q&A uses local FAISS semantic retrieval by default, but the embedding model must be installed/downloaded on the machine.
- Q&A falls back to TF-IDF retrieval if FAISS or sentence-transformers is unavailable; fallback semantic matching is more limited.
- API authentication, rate limiting, and user-level authorization are not implemented in the MVP.
- CORS should be restricted to trusted frontend origins before production deployment.
- Large audio files can take time to process and should be handled by a worker queue in production.
- Model outputs are prompt-based and should be reviewed for high-stakes business decisions.

## GitHub Safety

Do not commit:

```text
.env
.venv/
storage/
uploaded audio
cache folders
```

If a real API key, MongoDB URI, or password is ever pasted into chat or committed by mistake, rotate that credential immediately.

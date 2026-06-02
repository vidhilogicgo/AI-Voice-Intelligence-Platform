import asyncio
from collections import Counter
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import Settings
from core.usage_logging import log_model_fallback, log_model_usage
from schemas.audio import Insights, TranscriptSegment

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_INSIGHT_CHUNK_CHARS = 12000
MAX_ITEMS_PER_FIELD = 10


class GroqInsightError(Exception):
    pass


class InsightExtractionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def extract(self, transcript: list[TranscriptSegment]) -> Insights:
        if not transcript:
            return _empty_insights()

        if self.settings.groq_api_key:
            log_model_usage(
                provider="Groq",
                model=self.settings.groq_insight_model,
                purpose="business insight extraction",
                details=f"segments={len(transcript)}",
            )
            try:
                insights = await asyncio.to_thread(self._extract_with_groq, transcript)
            except GroqInsightError as exc:
                log_model_fallback(
                    provider="local",
                    model="extractive-keyword-insights",
                    purpose="business insight extraction",
                    reason=str(exc),
                )
                return _extract_fallback_insights(transcript)
            if insights is not None:
                return insights
            log_model_fallback(
                provider="local",
                model="extractive-keyword-insights",
                purpose="business insight extraction",
                reason="Groq returned no parseable insights",
            )
            return _extract_fallback_insights(transcript)

        log_model_usage(
            provider="local",
            model="extractive-keyword-insights",
            purpose="business insight extraction",
            mode="fallback",
            details="GROQ_API_KEY not set",
        )
        return _extract_fallback_insights(transcript)

    def _extract_with_groq(
        self,
        transcript: list[TranscriptSegment],
    ) -> Insights | None:
        transcript_text = _transcript_for_prompt(transcript)
        chunks = _chunk_text(transcript_text, MAX_INSIGHT_CHUNK_CHARS)
        if not chunks:
            return None

        partials: list[Insights] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = _build_insight_prompt(chunk, index, len(chunks))
            partial = self._request_insights(prompt)
            if partial is None:
                return None
            partials.append(partial)

        if len(partials) == 1:
            return partials[0]

        return self._request_insights(_build_final_insight_prompt(partials))

    def _request_insights(self, prompt: str) -> Insights | None:
        payload = {
            "model": self.settings.groq_insight_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract business insights from call transcripts for "
                        "sales, support, product, research, and management teams. "
                        "You infer useful business meaning from the conversation "
                        "without inventing unsupported facts. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = Request(
            GROQ_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "VoiceIntelligenceAPI/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=45) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise GroqInsightError(
                f"Groq HTTP {exc.code}: {_compact_error_body(exc)}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise GroqInsightError(f"Groq request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GroqInsightError("Groq returned invalid JSON") from exc

        content = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_insights_json(content)


def _build_insight_prompt(chunk: str, index: int, total: int) -> str:
    return (
        f"Extract business insights from transcript chunk {index} of {total}.\n"
        "Use only what is supported by the transcript, but convert conversation into useful business categories.\n"
        "Do not copy transcript lines as-is. Write each item as a clear insight or action-oriented takeaway.\n"
        "Keep each item concise and concrete.\n"
        "If a field has no evidence, return an empty list or 'Unknown' for sentiment/buying_intent.\n\n"
        "Fields:\n"
        "- pain_points: problems, frustrations, inefficiencies, or unmet needs expressed or clearly implied.\n"
        "- objections: concerns, hesitation, blockers, risks, or reasons not to proceed.\n"
        "- requirements: must-have needs, requested capabilities, constraints, or success criteria.\n"
        "- feature_requests: product improvements or new functionality requested or implied.\n"
        "- sentiment: concise overall tone with business context.\n"
        "- buying_intent: Low, Medium, High, or Unknown with a short reason.\n"
        "- action_items: next steps, owners, follow-ups, decisions requiring action, or items to investigate.\n\n"
        'Return JSON exactly like: {"pain_points":[],"objections":[],"requirements":[],"feature_requests":[],"sentiment":"...","buying_intent":"...","action_items":[]}\n\n'
        f"Transcript:\n{chunk}"
    )


def _build_final_insight_prompt(partials: list[Insights]) -> str:
    partial_text = "\n\n".join(
        f"Chunk {index}: {partial.model_dump_json()}"
        for index, partial in enumerate(partials, start=1)
    )
    return (
        "Merge these partial call insights into one final business insight object.\n"
        "Remove duplicates, preserve only evidence-backed points, and keep the output concise.\n"
        "The final object should read like manager-ready CRM/product/support notes, not transcript snippets.\n"
        'Return JSON exactly like: {"pain_points":[],"objections":[],"requirements":[],"feature_requests":[],"sentiment":"...","buying_intent":"...","action_items":[]}\n\n'
        f"Partial insights:\n{partial_text}"
    )


def _parse_insights_json(content: str) -> Insights | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return Insights(
        pain_points=_clean_list(payload.get("pain_points", [])),
        objections=_clean_list(payload.get("objections", [])),
        requirements=_clean_list(payload.get("requirements", [])),
        feature_requests=_clean_list(payload.get("feature_requests", [])),
        sentiment=_clean_text(str(payload.get("sentiment", "Unknown"))),
        buying_intent=_clean_text(str(payload.get("buying_intent", "Unknown"))),
        action_items=_clean_list(payload.get("action_items", [])),
    )


def _transcript_for_prompt(transcript: list[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{segment.start_time}-{segment.end_time}] {segment.speaker}: {segment.text}"
        for segment in transcript
        if segment.text.strip()
    )


def _chunk_text(text: str, max_chars: int) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0

    for line in lines:
        projected_length = current_length + len(line) + 1
        if current_lines and projected_length > max_chars:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_length = 0
        current_lines.append(line)
        current_length += len(line) + 1

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _extract_fallback_insights(transcript: list[TranscriptSegment]) -> Insights:
    sentences = _split_sentences(" ".join(segment.text for segment in transcript))
    ranked = _rank_sentences(sentences)
    points = [_clean_text(sentence).rstrip(".") for sentence in ranked[:5]]
    return Insights(
        pain_points=[],
        objections=[],
        requirements=points[:3],
        feature_requests=[],
        sentiment="Unknown",
        buying_intent="Unknown",
        action_items=points[3:5],
    )


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentence.split()) >= 4
    ]


def _rank_sentences(sentences: list[str]) -> list[str]:
    words = [
        word
        for sentence in sentences
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", sentence.lower())
        if len(word) > 3
    ]
    frequencies = Counter(words)
    return sorted(
        sentences,
        key=lambda sentence: sum(
            frequencies[word]
            for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", sentence.lower())
        ),
        reverse=True,
    )


def _empty_insights() -> Insights:
    return Insights(
        pain_points=[],
        objections=[],
        requirements=[],
        feature_requests=[],
        sentiment="Unknown",
        buying_intent="Unknown",
        action_items=[],
    )


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe(
        [_clean_text(str(item)).rstrip(".") for item in value if _clean_text(str(item))]
    )[:MAX_ITEMS_PER_FIELD]


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip("- ")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _read_error_body(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason or "No error details returned."
    return body[:500] if body else (exc.reason or "No error details returned.")


def _compact_error_body(exc: HTTPError) -> str:
    body = _read_error_body(exc)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", body).strip()[:240]

    error = payload.get("error", {})
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or "unknown_error"
        message = str(error.get("message") or "").strip()
        return f"{code} - {message[:200]}"

    return re.sub(r"\s+", " ", body).strip()[:240]

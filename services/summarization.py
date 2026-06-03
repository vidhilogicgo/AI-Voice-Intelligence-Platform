import asyncio
from collections import Counter
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import Settings
from core.usage_logging import log_model_fallback, log_model_usage
from schemas.audio import Summary, TranscriptSegment

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_SUMMARY_CHUNK_CHARS = 12000
MAX_KEY_POINTS = 8


class GroqSummarizationError(Exception):
    pass


class SummarizationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def summarize(self, transcript: list[TranscriptSegment]) -> Summary:
        if not transcript:
            return Summary(
                short="No transcript content was available for summarization.",
                detailed="The uploaded call did not produce transcript segments that could be summarized.",
                key_points=[],
            )

        if self.settings.groq_api_key:
            log_model_usage(
                provider="Groq",
                model=self.settings.groq_summary_model,
                purpose="call summarization",
                details=f"segments={len(transcript)}",
            )
            try:
                llm_summary = await self._summarize_with_groq(transcript)
            except GroqSummarizationError as exc:
                print(f"❌ [SUMMARIZATION] Groq summary model ({self.settings.groq_summary_model}) failed: {exc}. 📍 Fallback: using local extractive frequency model.")
                log_model_fallback(
                    provider="local",
                    model="extractive-frequency",
                    purpose="call summarization",
                    reason=str(exc),
                )
                return self._summarize_extractively(transcript)
            if llm_summary is not None:
                print(f"✅ [SUMMARIZATION] Groq summary model ({self.settings.groq_summary_model}) succeeded.")
                return llm_summary
            print(f"❌ [SUMMARIZATION] Groq summary model ({self.settings.groq_summary_model}) returned no parseable summary. 📍 Fallback: using local extractive frequency model.")
            log_model_fallback(
                provider="local",
                model="extractive-frequency",
                purpose="call summarization",
                reason="Groq returned no parseable summary",
            )
            return self._summarize_extractively(transcript)

        print("📍 [SUMMARIZATION] Groq API key not configured. Fallback: using local extractive frequency model.")
        log_model_usage(
            provider="local",
            model="extractive-frequency",
            purpose="call summarization",
            mode="fallback",
            details="GROQ_API_KEY not set",
        )
        return self._summarize_extractively(transcript)

    async def _summarize_with_groq(
        self,
        transcript: list[TranscriptSegment],
    ) -> Summary | None:
        return await asyncio.to_thread(self._request_groq_summary, transcript)

    def _request_groq_summary(
        self,
        transcript: list[TranscriptSegment],
    ) -> Summary | None:
        transcript_text = _transcript_for_prompt(transcript)
        chunks = _chunk_text(transcript_text, MAX_SUMMARY_CHUNK_CHARS)
        if not chunks:
            return None

        if len(chunks) == 1:
            return self._request_single_groq_summary(_build_summary_prompt(chunks[0]))

        partial_summaries: list[Summary] = []
        for index, chunk in enumerate(chunks, start=1):
            partial = self._request_single_groq_summary(
                _build_chunk_summary_prompt(chunk, index, len(chunks))
            )
            if partial is None:
                return None
            partial_summaries.append(partial)

        synthesis_prompt = _build_final_summary_prompt(partial_summaries)
        return self._request_single_groq_summary(synthesis_prompt)

    def _request_single_groq_summary(self, prompt: str) -> Summary | None:
        payload = {
            "model": self.settings.groq_summary_model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert call transcript analyst. Your job is to summarize call transcripts accurately "
                        "based on their actual content and context. "
                        "First, identify the nature of the conversation: Is it business-focused? Technical? Personal? "
                        "Educational? Customer support? Then summarize accordingly.\n\n"
                        "For BUSINESS conversations: Extract business insights, decisions, action items, pain points, "
                        "requirements, sentiment, and next steps. Focus on outcomes and value.\n"
                        "For OTHER topics: Provide clear, relevant summaries appropriate to the topic. If it's casual, "
                        "technical, educational, or personal—summarize the key themes and takeaways in that context.\n\n"
                        "You DO NOT summarize gibberish, incoherent rambling, or unclear fragments. "
                        "If a transcript lacks coherent discussion (ANY topic), respond with: "
                        '{"short":"No clear content captured","detailed":"The transcript did not contain coherent discussion","key_points":[]}. '
                        "For coherent transcripts: synthesize and paraphrase (do not copy lines verbatim). "
                        "Explain what happened, why it matters in context, and relevant next steps or conclusions.\n\n"
                        "Return only valid JSON with short, detailed, and key_points fields. "
                        "short = 1-2 lines that give a clear overview of the whole situation: "
                        "who/what the conversation is about, the main issue or goal, and the current outcome or next step if clear. "
                        "detailed = 2-4 paragraphs covering purpose, key topics, decisions/conclusions, and outcomes. "
                        "key_points = 4-8 specific, relevant takeaways or action items."
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
            raise GroqSummarizationError(
                f"Groq HTTP {exc.code}: {_compact_error_body(exc)}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise GroqSummarizationError(f"Groq request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GroqSummarizationError("Groq returned invalid JSON") from exc

        content = (
            response_payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_summary_json(content)

    def _summarize_extractively(self, transcript: list[TranscriptSegment]) -> Summary:
        full_text = _transcript_plain_text(transcript)
        sentences = _split_sentences(full_text)
        if not sentences:
            return Summary(
                short="The call transcript was too short to summarize reliably.",
                detailed="Only minimal transcript text was available, so no meaningful business summary could be generated.",
                key_points=[],
            )

        ranked_sentences = _rank_sentences(sentences)
        top_sentences = _restore_original_order(sentences, ranked_sentences[:5])
        key_points = [_as_key_point(sentence) for sentence in top_sentences]
        key_points = _dedupe_items(key_points)[:MAX_KEY_POINTS]

        short = _build_short_summary(top_sentences)
        detailed = _build_detailed_summary(transcript, top_sentences)
        return Summary(short=short, detailed=detailed, key_points=key_points)


def _build_summary_prompt(transcript_text: str) -> str:
    return (
        "Analyze this transcript and create a summary based on its actual content and nature.\n\n"
        "First, identify: Is this business-focused? Technical? Personal? Educational? Support? Or something else?\n\n"
        "Requirements for ALL topics:\n"
        "- short: 1-2 lines giving an overview of the whole situation: who/what the conversation is about, "
        "the main issue or goal, and the current outcome or next step if clear.\n"
        "- detailed: 2-4 paragraphs covering: (1) conversation purpose and context; (2) main topics discussed; "
        "(3) key decisions, conclusions, or takeaways; (4) outcomes, next steps, or action items (where relevant).\n"
        "- key_points: 4-8 specific, relevant conclusions or takeaways appropriate to the topic. Examples: decision made, "
        "issue identified, commitment, commitment, technical insight, learning point, or action item.\n\n"
        "Guidelines:\n"
        "- For BUSINESS conversations: Focus on business outcomes, decisions, commitments, pain points, requirements, sentiment, buying intent, and next steps.\n"
        "- For NON-BUSINESS conversations: Summarize the core themes, key discussion points, insights, and relevant conclusions in the conversation's actual context.\n"
        "- Capture only clear, coherent discussion. Ignore filler, false starts, side comments, and unclear fragments.\n"
        "- If the transcript is incoherent or lacks clear discussion (ANY topic), respond with "
        "short='No clear content captured', detailed='The transcript did not contain coherent discussion', key_points=[].\n"
        "- Do not invent details. Do not pad summaries with speculation.\n"
        "- Synthesize and paraphrase, don't copy transcript lines verbatim.\n"
        "- Explain who said what and why it matters in the context of the conversation.\n\n"
        'Return JSON exactly like: {"short":"...","detailed":"...","key_points":["..."]}\n\n'
        f"Transcript:\n{transcript_text}"
    )


def _build_chunk_summary_prompt(chunk: str, index: int, total: int) -> str:
    return (
        f"Summarize chunk {index} of {total} from a longer transcript.\n\n"
        "First, identify the conversation's nature: business, technical, personal, educational, support, or other?\n"
        "Then extract what happened in this part—key topics, conclusions, insights, commitments, or decisions relevant to the conversation type.\n\n"
        "Task:\n"
        "- short: 1-2 lines summarizing the situation covered in this chunk, not just a copied sentence.\n"
        "- For BUSINESS sections: Capture speaker intent, needs, concerns, decisions, and follow-ups.\n"
        "- For NON-BUSINESS sections: Capture the relevant themes, insights, or conclusions for that type of discussion.\n"
        "- Write analytical notes, not transcript copies.\n"
        "- Skip filler, unclear rambling, or incoherent fragments. Focus only on clear, coherent discussion.\n"
        "- If this chunk is incoherent or contains no clear content, respond with short='No clear content', detailed='No coherent discussion', key_points=[].\n"
        'Return JSON exactly like: {"short":"...","detailed":"...","key_points":["..."]}\n\n'
        f"Transcript chunk:\n{chunk}"
    )


def _build_final_summary_prompt(partial_summaries: list[Summary]) -> str:
    summaries = "\n\n".join(
        (
            f"Chunk {index}\n"
            f"Short: {summary.short}\n"
            f"Detailed: {summary.detailed}\n"
            f"Key points: {', '.join(summary.key_points)}"
        )
        for index, summary in enumerate(partial_summaries, start=1)
    )
    return (
        "Create the final comprehensive summary for the entire transcript from these chunk summaries.\n\n"
        "First, identify the overall nature of the conversation: Is it business-focused? Technical? Personal? Educational? Support? Or other?\n\n"
        "Then synthesize accordingly:\n"
        "1. If most chunks contain 'No clear content' or 'No coherent discussion', respond with short='No clear content captured', detailed='The transcript did not contain coherent discussion', key_points=[].\n"
        "2. Otherwise, merge chunk summaries into a cohesive narrative:\n"
        "   - short must be 1-2 lines giving the overall situation across the whole transcript, including the main issue/goal and outcome or next step if clear.\n"
        "   - Describe what happened across the full conversation (not just list chunks).\n"
        "   - Merge duplicate or similar points; preserve chronology where useful.\n"
        "   - For BUSINESS: Highlight business outcomes, decisions, commitments, or unresolved issues.\n"
        "   - For NON-BUSINESS: Highlight core themes, key insights, conclusions, or relevant next steps in that context.\n"
        "   - Do not invent connections not supported by the chunks.\n"
        "   - The summary should read professionally and make sense for the conversation's actual purpose.\n\n"
        'Return JSON exactly like: {"short":"...","detailed":"...","key_points":["..."]}\n\n'
        f"Chunk summaries:\n{summaries}"
    )


def _transcript_for_prompt(transcript: list[TranscriptSegment]) -> str:
    lines = [
        f"[{segment.start_time}-{segment.end_time}] {segment.speaker}: {segment.text}"
        for segment in transcript
        if segment.text.strip()
    ]
    return "\n".join(lines)


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


def _parse_summary_json(content: str) -> Summary | None:
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

    short = _clean_summary_text(str(payload.get("short", "")))
    detailed = _clean_summary_text(str(payload.get("detailed", "")))
    key_points = payload.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []

    cleaned_points = [
        _clean_summary_text(str(point))
        for point in key_points
        if _clean_summary_text(str(point))
    ]
    if not short or not detailed:
        return None

    return Summary(
        short=short,
        detailed=detailed,
        key_points=_dedupe_items(cleaned_points)[:MAX_KEY_POINTS],
    )


def _transcript_plain_text(transcript: list[TranscriptSegment]) -> str:
    return " ".join(segment.text.strip() for segment in transcript if segment.text.strip())


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if len(sentence.split()) >= 4]


def _rank_sentences(sentences: list[str]) -> list[str]:
    words = [
        word
        for sentence in sentences
        for word in _words(sentence)
        if len(word) > 3
    ]
    frequencies = Counter(words)

    def score(sentence: str) -> float:
        sentence_words = _words(sentence)
        if not sentence_words:
            return 0
        frequency_score = sum(frequencies[word] for word in sentence_words)
        length_penalty = max(1, len(sentence_words) / 24)
        return frequency_score / length_penalty

    return sorted(sentences, key=score, reverse=True)


def _restore_original_order(sentences: list[str], selected: list[str]) -> list[str]:
    selected_set = set(selected)
    return [sentence for sentence in sentences if sentence in selected_set]


def _build_short_summary(sentences: list[str]) -> str:
    if not sentences:
        return "The transcript provides limited context, but it indicates a conversation occurred without enough clear detail to summarize the full situation."
    selected = sentences[:1]
    return _clean_summary_text(" ".join(selected))


def _build_detailed_summary(
    transcript: list[TranscriptSegment],
    sentences: list[str],
) -> str:
    duration = _format_duration(transcript)
    discussion = " ".join(sentences[:5])
    if not discussion:
        discussion = _transcript_plain_text(transcript)
    discussion = _clean_summary_text(discussion)
    return (
        f"The call lasted approximately {duration} based on the transcript timestamps. "
        f"The main discussion centered on: {discussion}"
    )


def _as_key_point(sentence: str) -> str:
    point = _clean_summary_text(sentence)
    return point.rstrip(".")


def _format_duration(transcript: list[TranscriptSegment]) -> str:
    end_seconds = max((segment.end_seconds for segment in transcript), default=0.0)
    total_seconds = max(0, int(round(end_seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes == 0:
        return f"{seconds} seconds"
    if seconds == 0:
        return f"{minutes} minutes"
    return f"{minutes} minutes {seconds} seconds"


def _words(sentence: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]+", sentence.lower())


def _clean_summary_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("- ")
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _dedupe_items(items: list[str]) -> list[str]:
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

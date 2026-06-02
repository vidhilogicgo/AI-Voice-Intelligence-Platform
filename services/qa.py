import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import Settings, get_settings
from core.usage_logging import log_model_fallback, log_model_usage
from schemas.audio import AskResponse, TranscriptSegment

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
MIN_RELEVANCE_SCORE = 0.05


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    segments: tuple[TranscriptSegment, ...]
    vector: dict[str, float]


class TranscriptVectorStore:
    """Small local vector store for one transcript using sparse TF-IDF vectors."""

    def __init__(self, chunks: list[TranscriptChunk], idf: dict[str, float]) -> None:
        self.chunks = chunks
        self.idf = idf

    @classmethod
    def from_transcript(
        cls,
        transcript: list[TranscriptSegment],
        max_chars: int,
        overlap_segments: int,
    ) -> "TranscriptVectorStore":
        chunk_segments = _chunk_transcript_segments(
            transcript=transcript,
            max_chars=max_chars,
            overlap_segments=overlap_segments,
        )
        chunk_texts = [_format_segments_for_context(segments) for segments in chunk_segments]
        idf = _build_idf(chunk_texts)
        chunks = [
            TranscriptChunk(
                text=text,
                segments=tuple(segments),
                vector=_embed_text(text, idf),
            )
            for text, segments in zip(chunk_texts, chunk_segments)
            if text.strip()
        ]
        return cls(chunks=chunks, idf=idf)

    def search(self, query: str, top_k: int) -> list[tuple[TranscriptChunk, float]]:
        query_vector = _embed_text(query, self.idf)
        if not query_vector:
            return []

        scored = [
            (chunk, _cosine_similarity(query_vector, chunk.vector))
            for chunk in self.chunks
        ]
        scored = [(chunk, score) for chunk, score in scored if score > 0]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class TranscriptQAService:
    _store_cache: dict[str, TranscriptVectorStore] = {}

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def answer(
        self,
        audio_id: str,
        question: str,
        transcript: list[TranscriptSegment],
    ) -> AskResponse:
        if not transcript:
            return AskResponse(
                audio_id=audio_id,
                question=question,
                answer="The transcript is empty, so there is not enough evidence to answer.",
                sources=[],
            )

        store = self._get_store(audio_id, transcript)
        retrieval_query = _expanded_query(question)
        matches = store.search(retrieval_query, top_k=self.settings.qa_retrieval_top_k)
        relevant_matches = [
            match for match in matches if match[1] >= MIN_RELEVANCE_SCORE
        ]
        selected_matches = relevant_matches or matches[:1]

        if not selected_matches:
            return AskResponse(
                audio_id=audio_id,
                question=question,
                answer=(
                    "The transcript does not contain enough relevant information "
                    "to answer that question."
                ),
                sources=[],
            )

        sources = _dedupe_segments(
            segment
            for chunk, _ in selected_matches
            for segment in chunk.segments
        )

        answer = None
        if self.settings.groq_api_key:
            try:
                answer = await asyncio.to_thread(
                    self._answer_with_groq,
                    question,
                    [chunk for chunk, _ in selected_matches],
                )
            except Exception as exc:
                log_model_fallback(
                    provider="local",
                    model="tf-idf-retrieval-extractive-answering",
                    purpose="transcript question answering",
                    reason=f"Groq answer generation failed unexpectedly: {exc}",
                )

        if answer is None:
            log_model_usage(
                provider="local",
                model="tf-idf-retrieval-extractive-answering",
                purpose="transcript question answering",
                mode="fallback",
                details=f"audio_id={audio_id} sources={len(sources)}",
            )
            answer = _answer_extractively(question, [chunk for chunk, _ in selected_matches])

        return AskResponse(
            audio_id=audio_id,
            question=question,
            answer=answer,
            sources=sources,
        )

    def _get_store(
        self,
        audio_id: str,
        transcript: list[TranscriptSegment],
    ) -> TranscriptVectorStore:
        signature = _transcript_signature(transcript)
        cache_key = f"{audio_id}:{signature}"
        if cache_key not in self._store_cache:
            log_model_usage(
                provider="local",
                model="tf-idf-sparse-vector-store",
                purpose="transcript retrieval index",
                details=f"audio_id={audio_id} segments={len(transcript)}",
            )
            self._store_cache[cache_key] = TranscriptVectorStore.from_transcript(
                transcript=transcript,
                max_chars=max(500, self.settings.qa_chunk_max_chars),
                overlap_segments=max(0, self.settings.qa_chunk_overlap_segments),
            )
        return self._store_cache[cache_key]

    def _answer_with_groq(
        self,
        question: str,
        chunks: list[TranscriptChunk],
    ) -> str | None:
        context = "\n\n".join(
            f"Excerpt {index}:\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )
        log_model_usage(
            provider="Groq",
            model=self.settings.groq_qa_model,
            purpose="transcript question answering",
            details=f"retrieved_chunks={len(chunks)}",
        )
        payload = {
            "model": self.settings.groq_qa_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions about uploaded call transcripts. "
                        "Use only the provided transcript excerpts as evidence. "
                        "Do not use outside knowledge, summaries, or assumptions. "
                        "If the excerpts do not answer the question, say that the "
                        "transcript does not provide enough evidence. Return only "
                        "valid JSON with an answer field."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Transcript excerpts:\n{context}\n\n"
                        'Return JSON exactly like: {"answer":"..."}'
                    ),
                },
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
            log_model_fallback(
                provider="local",
                model="tf-idf-retrieval-extractive-answering",
                purpose="transcript question answering",
                reason=f"Groq HTTP {exc.code}: {_compact_error_body(exc)}",
            )
            return None
        except (URLError, TimeoutError) as exc:
            log_model_fallback(
                provider="local",
                model="tf-idf-retrieval-extractive-answering",
                purpose="transcript question answering",
                reason=f"Groq request failed: {exc}",
            )
            return None
        except json.JSONDecodeError:
            log_model_fallback(
                provider="local",
                model="tf-idf-retrieval-extractive-answering",
                purpose="transcript question answering",
                reason="Groq returned invalid JSON",
            )
            return None

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            log_model_fallback(
                provider="local",
                model="tf-idf-retrieval-extractive-answering",
                purpose="transcript question answering",
                reason="Groq response did not include choices",
            )
            return None

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return _parse_answer_json(content)


def _chunk_transcript_segments(
    transcript: list[TranscriptSegment],
    max_chars: int,
    overlap_segments: int,
) -> list[list[TranscriptSegment]]:
    clean_segments = [segment for segment in transcript if segment.text.strip()]
    if not clean_segments:
        return []

    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_length = 0

    for segment in clean_segments:
        line = _format_segment_for_context(segment)
        projected_length = current_length + len(line) + 1
        if current and projected_length > max_chars:
            chunks.append(current)
            current = current[-overlap_segments:] if overlap_segments else []
            current_length = sum(
                len(_format_segment_for_context(item)) + 1 for item in current
            )

        current.append(segment)
        current_length += len(line) + 1

    if current:
        chunks.append(current)

    return chunks


def _format_segments_for_context(segments: list[TranscriptSegment]) -> str:
    return "\n".join(_format_segment_for_context(segment) for segment in segments)


def _format_segment_for_context(segment: TranscriptSegment) -> str:
    return (
        f"[{segment.start_time}-{segment.end_time}] "
        f"{segment.speaker}: {segment.text.strip()}"
    )


def _build_idf(documents: list[str]) -> dict[str, float]:
    doc_tokens = [set(_tokens(document)) for document in documents]
    document_count = max(1, len(doc_tokens))
    terms = sorted({token for tokens in doc_tokens for token in tokens})
    return {
        term: math.log((1 + document_count) / (1 + _document_frequency(term, doc_tokens))) + 1
        for term in terms
    }


def _document_frequency(term: str, doc_tokens: list[set[str]]) -> int:
    return sum(1 for tokens in doc_tokens if term in tokens)


def _embed_text(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(_tokens(text))
    weighted = {
        token: count * idf[token]
        for token, count in counts.items()
        if token in idf
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return {}
    return {token: value / norm for token, value in weighted.items()}


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _answer_extractively(question: str, chunks: list[TranscriptChunk]) -> str:
    sentences = _sentences_from_chunks(chunks)
    if not sentences:
        return "The transcript does not contain enough relevant information to answer that question."

    intent_keywords = _intent_keywords(question)
    ranked = _rank_sentences(question, sentences, intent_keywords)
    selected = ranked[:3]
    if not selected:
        return "The transcript does not contain enough relevant information to answer that question."

    if _asks_about_interest(question):
        return _answer_interest(selected)

    if _asks_about_objections(question):
        objection_sentences = [
            sentence
            for sentence in ranked
            if _contains_any(sentence, OBJECTION_KEYWORDS)
        ][:3]
        if not objection_sentences:
            return "The retrieved transcript excerpts do not show a clear objection."
        return "Objections raised: " + " ".join(objection_sentences)

    if _asks_about_follow_up(question):
        follow_up_sentences = [
            sentence
            for sentence in ranked
            if _contains_any(sentence, FOLLOW_UP_KEYWORDS)
        ][:3]
        if not follow_up_sentences:
            return "The retrieved transcript excerpts do not state a clear follow-up."
        return "Follow-up required: " + " ".join(follow_up_sentences)

    if _asks_about_concern(question):
        concern_sentences = [
            sentence
            for sentence in ranked
            if _contains_any(sentence, CONCERN_KEYWORDS)
        ][:3]
        selected = concern_sentences or selected
        return "The customer's main concern appears to be: " + " ".join(selected)

    return "Based on the transcript excerpts: " + " ".join(selected)


def _sentences_from_chunks(chunks: list[TranscriptChunk]) -> list[str]:
    sentences: list[str] = []
    for chunk in chunks:
        for segment in chunk.segments:
            prefix = f"{segment.speaker} said "
            for sentence in _split_sentences(segment.text):
                sentences.append(prefix + sentence)
    return _dedupe_text(sentences)


def _rank_sentences(
    question: str,
    sentences: list[str],
    intent_keywords: set[str],
) -> list[str]:
    question_tokens = set(_tokens(question))

    def score(sentence: str) -> float:
        sentence_tokens = set(_tokens(sentence))
        overlap = len(question_tokens & sentence_tokens)
        intent_overlap = len(intent_keywords & sentence_tokens)
        return overlap + (intent_overlap * 2.5)

    return sorted(sentences, key=score, reverse=True)


def _intent_keywords(question: str) -> set[str]:
    lowered = question.lower()
    if "objection" in lowered or "object" in lowered:
        return OBJECTION_KEYWORDS
    if "follow" in lowered or "next step" in lowered or "required" in lowered:
        return FOLLOW_UP_KEYWORDS
    if "interest" in lowered or "interested" in lowered or "buying" in lowered:
        return INTEREST_KEYWORDS
    if "concern" in lowered or "main" in lowered or "pain" in lowered:
        return CONCERN_KEYWORDS
    return set()


def _expanded_query(question: str) -> str:
    keywords = sorted(_intent_keywords(question))
    if not keywords:
        return question
    return f"{question} {' '.join(keywords)}"


def _answer_interest(sentences: list[str]) -> str:
    positive = [sentence for sentence in sentences if _contains_any(sentence, INTEREST_KEYWORDS)]
    negative = [sentence for sentence in sentences if _contains_any(sentence, LOW_INTEREST_KEYWORDS)]
    if positive and not negative:
        return "The customer showed interest based on: " + " ".join(positive[:3])
    if negative and not positive:
        return "The transcript suggests low or unclear interest based on: " + " ".join(negative[:3])
    return "The transcript gives mixed or limited evidence of interest: " + " ".join(sentences[:3])


def _parse_answer_json(content: str) -> str | None:
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

    if not isinstance(payload, dict):
        return None

    answer = _clean_text(str(payload.get("answer", "")))
    return answer or None


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if len(sentence.split()) >= 3
    ]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _contains_any(text: str, keywords: set[str]) -> bool:
    tokens = set(_tokens(text))
    return bool(tokens & keywords)


def _asks_about_interest(question: str) -> bool:
    return bool({"interest", "interested", "buying", "intent"} & set(_tokens(question)))


def _asks_about_objections(question: str) -> bool:
    return bool({"objection", "objections", "objected", "raised"} & set(_tokens(question)))


def _asks_about_follow_up(question: str) -> bool:
    lowered = question.lower()
    return "follow" in lowered or "next step" in lowered or "required" in lowered


def _asks_about_concern(question: str) -> bool:
    return bool({"concern", "concerns", "pain", "issue", "problem"} & set(_tokens(question)))


def _dedupe_segments(segments: object) -> list[TranscriptSegment]:
    deduped: list[TranscriptSegment] = []
    seen: set[tuple[int, float, float, str]] = set()
    for segment in segments:
        if not isinstance(segment, TranscriptSegment):
            continue
        key = (
            segment.segment_id,
            segment.start_seconds,
            segment.end_seconds,
            segment.text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(segment)
    return deduped


def _dedupe_text(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip("- ")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


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


def _transcript_signature(transcript: list[TranscriptSegment]) -> str:
    segment_count = len(transcript)
    text_length = sum(len(segment.text) for segment in transcript)
    last_end = max((segment.end_seconds for segment in transcript), default=0.0)
    return f"{segment_count}:{text_length}:{last_end:.3f}"


STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "ask",
    "asked",
    "based",
    "been",
    "but",
    "call",
    "can",
    "could",
    "customer",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "main",
    "our",
    "out",
    "question",
    "that",
    "the",
    "their",
    "there",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}

CONCERN_KEYWORDS = {
    "blocker",
    "challenge",
    "concern",
    "concerns",
    "cost",
    "delay",
    "difficult",
    "expensive",
    "issue",
    "pain",
    "problem",
    "risk",
    "worried",
}

OBJECTION_KEYWORDS = {
    "budget",
    "concern",
    "concerns",
    "cost",
    "expensive",
    "hesitant",
    "objection",
    "objections",
    "pricing",
    "risk",
    "too",
    "wait",
}

FOLLOW_UP_KEYWORDS = {
    "action",
    "call",
    "demo",
    "email",
    "follow",
    "meeting",
    "next",
    "schedule",
    "send",
    "share",
    "step",
    "timeline",
}

INTEREST_KEYWORDS = {
    "agree",
    "demo",
    "evaluate",
    "interested",
    "like",
    "need",
    "pilot",
    "schedule",
    "trial",
    "want",
    "yes",
}

LOW_INTEREST_KEYWORDS = {
    "delay",
    "hesitant",
    "later",
    "maybe",
    "no",
    "not",
    "pause",
    "wait",
}

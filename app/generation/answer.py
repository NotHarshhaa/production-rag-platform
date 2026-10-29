"""Answer generation service and pluggable LLM providers.

``AnswerService`` glues retrieval, prompting, and the LLM together and
enforces grounding: when retrieval returns nothing useful, the service says
so instead of letting the model improvise. Citations are parsed from the
model's [n] markers and resolved against the chunks actually retrieved.

LLM providers:
- ``StubLLM`` — deterministic extractive answers, for tests and offline evals.
- ``OpenAIChatLLM`` — any OpenAI-compatible chat completions endpoint.
"""

import hashlib
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from app.core.config import LLM_OPENAI, LLM_STUB, Settings
from app.core.logging import get_logger
from app.evaluation.metrics import faithfulness_estimate
from app.generation.prompts import PROMPT_VERSION, build_context, build_messages
from app.ingestion.chunking import estimate_tokens, tokenize
from app.observability.metrics import LLM_COST, LLM_ERRORS, LLM_LATENCY, LLM_TOKENS, QUERY_CACHE
from app.retrieval.hybrid_search import RetrievedChunk
from app.retrieval.reranker import Reranker

logger = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LLMError(RuntimeError):
    """Raised when the LLM provider fails after retries."""


def estimate_tokens_pair(*texts: str) -> int:
    return sum(estimate_tokens(text) for text in texts)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@dataclass
class Citation:
    document_id: str
    document_name: str
    page: int
    chunk_id: str
    score: float


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    insufficient_context: bool = False
    groundedness_score: float = 1.0
    retrieval_mode: str = "hybrid"
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = PROMPT_VERSION
    reranked: bool = False
    cached: bool = False


class QueryCache:
    """Small TTL cache for answer results, invalidated by corpus changes.

    Entries remember the vector store revision at insert time; any document
    indexing or deletion bumps the revision and invalidates the whole cache,
    so answers never go stale after the corpus changes.
    """

    def __init__(self, ttl_seconds: int, max_entries: int = 256) -> None:
        self.ttl_seconds = max(0, ttl_seconds)
        self.max_entries = max_entries
        self._entries: dict[str, tuple[Any, float, int]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0

    @staticmethod
    def make_key(
        question: str,
        top_k: int | None,
        metadata_filter: dict | None,
        mode: str | None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        material = json.dumps(
            {
                "question": question.strip().lower(),
                "top_k": top_k,
                "filter": metadata_filter,
                "mode": mode,
                "history": conversation_history,
            },
            sort_keys=True,
        )
        return hashlib.sha1(material.encode("utf-8")).hexdigest()  # noqa: S324 - not security

    def get(self, key: str, current_revision: int) -> Any | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            value, expires_at, revision = entry
            if expires_at <= now or revision != current_revision:
                del self._entries[key]
                return None
            return value

    def put(self, key: str, value: Any, revision: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            if len(self._entries) >= self.max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k][1])
                del self._entries[oldest]
            self._entries[key] = (value, time.monotonic() + self.ttl_seconds, revision)


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> Iterator[str]: ...


class StubLLM(LLMProvider):
    """Deterministic extractive "LLM" used by tests and offline evaluation.

    Scores context sentences by keyword overlap with the question and joins
    the best ones, with a citation marker per used passage.
    """

    name = "stub"
    model = "stub-extractive"

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024
    ) -> LLMResponse:
        question, context = self._split_messages(messages)
        answer = self._extractive_answer(question, context)
        return LLMResponse(
            text=answer,
            prompt_tokens=estimate_tokens_pair(question, context),
            completion_tokens=estimate_tokens(answer),
            model=self.model,
        )

    def stream(
        self, messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 1024
    ) -> Iterator[str]:
        response = self.generate(messages, temperature, max_tokens)
        for word in response.text.split(" "):
            yield word + " "

    @staticmethod
    def _split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
        question = ""
        context = ""
        for message in messages:
            content = message.get("content", "")
            if message.get("role") != "user":
                continue
            if content.startswith("Context passages:"):
                # build_messages() puts context and question in one message.
                marker = content.find("\nQuestion: ")
                if marker != -1:
                    question = content[marker:].removeprefix("\nQuestion: ").split("\n")[0].strip()
                    context = content[:marker]
                else:
                    context = content
            elif content:
                question = content.removeprefix("Question: ").strip()
        return question, context

    def _extractive_answer(self, question: str, context: str) -> str:
        query_tokens = set(tokenize(question))
        passages = [
            passage
            for passage in re.split(r"\n(?=\[\d+\] )", context)
            if re.match(r"\s*\[\d+\]", passage)
        ]
        candidates: list[tuple[int, str]] = []
        for passage_index, passage in enumerate(passages):
            header, _, body = passage.partition("\n")
            match = re.match(r"\[(\d+)\]", header)
            citation_number = int(match.group(1)) if match else passage_index + 1
            for sentence in _SENTENCE_SPLIT.split(body):
                overlap = len(query_tokens & set(tokenize(sentence)))
                if overlap > 0:
                    candidates.append((-overlap, f"{sentence.strip()} [{citation_number}]"))

        if not candidates:
            first_passage = passages[0].partition("\n")[2] if passages else ""
            first_sentence = _SENTENCE_SPLIT.split(first_passage)[0] if first_passage else ""
            if not first_sentence:
                return "I could not find relevant information in the indexed documents."
            return f"{first_sentence.strip()} [1]"

        candidates.sort(key=lambda item: (item[0], len(item[1])))
        chosen = [text for _score, text in candidates[:3]]
        return " ".join(chosen)


class OpenAIChatLLM(LLMProvider):
    """Chat completions against any OpenAI-compatible API, with retries."""

    def __init__(self, settings: Settings) -> None:
        self.name = "openai"
        self.model = settings.llm_model
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"
        return headers

    def _payload(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int, stream: bool
    ) -> dict:
        return {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def generate(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> LLMResponse:
        import httpx

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=self._payload(messages, temperature, max_tokens, stream=False),
                    timeout=self._settings.llm_timeout_seconds,
                )
                if response.status_code >= 400:
                    raise LLMError(
                        f"LLM request failed with status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                body = response.json()
                usage = body.get("usage") or {}
                LLM_LATENCY.labels(provider=self.name).observe(time.perf_counter() - started)
                return LLMResponse(
                    text=body["choices"][0]["message"]["content"] or "",
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    model=body.get("model", self.model),
                )
            except (httpx.TimeoutException, httpx.TransportError, LLMError) as exc:
                last_error = exc
                if attempt < self._settings.llm_max_retries:
                    time.sleep(0.5 * (2**attempt))

        LLM_ERRORS.labels(provider=self.name).inc()
        raise LLMError(f"LLM request failed after retries: {last_error}")

    def stream(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> Iterator[str]:
        import httpx

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            with httpx.stream(
                "POST",
                url,
                headers=self._headers(),
                json=self._payload(messages, temperature, max_tokens, stream=True),
                timeout=self._settings.llm_timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise LLMError(
                        f"LLM stream failed with status {response.status_code}: {body[:200]}"
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except LLMError:
            LLM_ERRORS.labels(provider=self.name).inc()
            raise
        except Exception as exc:
            LLM_ERRORS.labels(provider=self.name).inc()
            raise LLMError(f"LLM streaming failed: {exc}") from exc


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == LLM_STUB:
        return StubLLM()
    if provider == LLM_OPENAI:
        return OpenAIChatLLM(settings)
    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Supported: {LLM_STUB}, {LLM_OPENAI}"
    )


def parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve [n] markers in the answer against the retrieved chunks."""

    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(chunks) and number not in seen:
            seen.append(number)
    citations = []
    for number in seen:
        chunk = chunks[number - 1]
        citations.append(
            Citation(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
        )
    return citations


def _contextual_search_query(question: str, history: list[dict[str, str]] | None) -> str:
    """Expand search query with context from previous user turn if needed."""
    if not history:
        return question
    last_user_turn = None
    for turn in reversed(history):
        if turn.get("role") == "user" and turn.get("content"):
            last_user_turn = turn.get("content", "").strip()
            break
    if not last_user_turn:
        return question
    pronouns = {
        "it", "this", "that", "these", "those", "they", "its", "their", "second",
        "first", "third", "former", "latter", "again", "elaborate", "expand",
    }
    words = set(re.findall(r"\b\w+\b", question.lower()))
    if len(words) <= 5 or bool(words & pronouns):
        return f"{last_user_turn} {question}"
    return question


class AnswerService:
    """Retrieval-augmented answering with enforced grounding and citations."""

    def __init__(
        self,
        retriever,
        reranker: Reranker,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.settings = settings
        self.cache = QueryCache(settings.query_cache_ttl_seconds)

    def _cache_key(
        self,
        question: str,
        top_k: int | None,
        metadata_filter: dict[str, Any] | None,
        mode: str | None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        return self.cache.make_key(question, top_k, metadata_filter, mode, conversation_history)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (
            prompt_tokens / 1000 * self.settings.llm_cost_prompt_per_1k
            + completion_tokens / 1000 * self.settings.llm_cost_completion_per_1k
        )
        return round(cost, 6)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        mode: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AnswerResult:
        revision = self.retriever.vector_store.get_revision()
        cache_key = self._cache_key(question, top_k, metadata_filter, mode, conversation_history)
        cached_result = self.cache.get(cache_key, revision)
        if cached_result is not None:
            QUERY_CACHE.labels(result="hit").inc()
            return replace(cached_result, cached=True)
        QUERY_CACHE.labels(result="miss").inc()

        started = time.perf_counter()
        search_query = _contextual_search_query(question, conversation_history)
        chunks = self.retriever.retrieve(
            search_query, top_k=top_k, metadata_filter=metadata_filter, mode=mode
        )
        retrieval_latency_ms = (time.perf_counter() - started) * 1000

        reranked = False
        if chunks and self.reranker.enabled:
            limit = top_k or self.settings.retrieval_top_k
            chunks = self.reranker.rerank(search_query, chunks, limit)
            reranked = True

        if not chunks:
            total_ms = (time.perf_counter() - started) * 1000
            result = AnswerResult(
                question=question,
                answer=(
                    "I could not find relevant information in the indexed documents "
                    "to answer this question."
                ),
                citations=[],
                chunks=[],
                insufficient_context=True,
                retrieval_mode=mode or "hybrid",
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                total_latency_ms=round(total_ms, 2),
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": estimate_tokens(question) // 2,
                    "total_tokens": estimate_tokens(question) // 2,
                    "cost_usd": 0.0,
                    "model": self.llm.model,
                },
                reranked=reranked,
            )
            self.cache.put(cache_key, result, revision)
            return result

        context = build_context(chunks)
        messages = build_messages(question, context, conversation_history=conversation_history)
        llm_started = time.perf_counter()
        try:
            response = self.llm.generate(
                messages,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )
        except LLMError:
            raise
        finally:
            LLM_LATENCY.labels(provider=self.llm.name).observe(time.perf_counter() - llm_started)
        LLM_TOKENS.labels(type="prompt").inc(response.prompt_tokens)
        LLM_TOKENS.labels(type="completion").inc(response.completion_tokens)

        cost = self._estimate_cost(response.prompt_tokens, response.completion_tokens)
        LLM_COST.inc(cost)

        citations = parse_citations(response.text, chunks)
        groundedness = round(faithfulness_estimate(response.text, [c.text for c in chunks]), 4)
        if not citations:
            # The model cited nothing; surface the top chunk so answers always
            # point at at least one source.
            top = chunks[0]
            citations = [
                Citation(
                    document_id=top.document_id,
                    document_name=top.document_name,
                    page=top.page,
                    chunk_id=top.chunk_id,
                    score=top.score,
                )
            ]

        total_ms = (time.perf_counter() - started) * 1000
        result = AnswerResult(
            question=question,
            answer=response.text,
            citations=citations,
            chunks=chunks,
            insufficient_context=False,
            groundedness_score=groundedness,
            retrieval_mode=mode or ("hybrid" if self.settings.hybrid_retrieval else "semantic"),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            total_latency_ms=round(total_ms, 2),
            usage={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.prompt_tokens + response.completion_tokens,
                "cost_usd": cost,
                "model": response.model or self.llm.model,
            },
            reranked=reranked,
        )
        self.cache.put(cache_key, result, revision)
        return result

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        mode: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield SSE-ready event dicts: retrieval, delta*, then done (or error)."""

        started = time.perf_counter()
        search_query = _contextual_search_query(question, conversation_history)
        chunks = self.retriever.retrieve(
            search_query, top_k=top_k, metadata_filter=metadata_filter, mode=mode
        )
        yield {
            "type": "retrieval",
            "data": {
                "chunks_returned": len(chunks),
                "mode": mode or ("hybrid" if self.settings.hybrid_retrieval else "semantic"),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page": chunk.page,
                        "score": chunk.score,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            },
        }

        if not chunks:
            yield {
                "type": "done",
                "data": {
                    "answer": (
                        "I could not find relevant information in the indexed documents "
                        "to answer this question."
                    ),
                    "citations": [],
                    "insufficient_context": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            }
            return

        context = build_context(chunks)
        messages = build_messages(question, context, conversation_history=conversation_history)
        collected: list[str] = []
        try:
            for delta in self.llm.stream(
                messages,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ):
                collected.append(delta)
                yield {"type": "delta", "data": {"text": delta}}
        except LLMError as exc:
            yield {"type": "error", "data": {"detail": str(exc)}}
            return

        answer = "".join(collected)
        citations = parse_citations(answer, chunks)
        yield {
            "type": "done",
            "data": {
                "answer": answer,
                "citations": [
                    {
                        "document_id": citation.document_id,
                        "document_name": citation.document_name,
                        "page": citation.page,
                        "chunk_id": citation.chunk_id,
                        "score": citation.score,
                    }
                    for citation in citations
                ],
                "insufficient_context": False,
                "groundedness_score": round(
                    faithfulness_estimate(answer, [c.text for c in chunks]), 4
                ),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }

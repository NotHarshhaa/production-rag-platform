"""Grounded question answering: POST /query and POST /query/stream."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.auth import Tenant
from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import (
    CitationOut,
    DebugScoreItem,
    QueryRequest,
    QueryResponse,
    RetrievalInfo,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    RetrievedChunkOut,
    UsageInfo,
)
from app.generation.answer import AnswerResult, LLMError
from app.storage.database import Conversation

router = APIRouter(tags=["query"])

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _build_filter(tenant: str, document_ids: list[str] | None) -> dict:
    # Tenant scoping is enforced server-side, never taken from the request.
    metadata_filter: dict = {"tenant_id": tenant}
    if document_ids:
        metadata_filter["document_id"] = document_ids
    return metadata_filter


def _resolve_conversation(
    container: ServiceContainer, tenant: str, request: QueryRequest
) -> tuple[Conversation | None, list[dict[str, str]] | None]:
    """Return the conversation to persist into (or None) and the history to use.

    With ``conversation_id`` the stored history is authoritative unless the
    client also sent explicit ``conversation_history`` turns.
    """

    history_dicts = (
        [m.model_dump() for m in request.conversation_history]
        if request.conversation_history
        else None
    )
    if not request.conversation_id:
        return None, history_dicts

    conversation = container.conversations.get(request.conversation_id, tenant)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {request.conversation_id} not found",
        )
    if history_dicts is not None:
        return conversation, history_dicts

    max_turns = container.settings.conversation_max_turns
    limit = max_turns * 2 if max_turns > 0 else None
    stored = container.conversations.list_messages(conversation.id, limit=limit)
    stored_history = [{"role": m.role, "content": m.content} for m in stored]
    return conversation, (stored_history or None)


def _persist_turn(
    container: ServiceContainer,
    conversation: Conversation,
    question: str,
    answer: str,
    citations: list,
) -> None:
    """Append the question and the answer to the stored conversation."""

    citation_dicts = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page": c.page,
            "chunk_id": c.chunk_id,
            "score": c.score,
        }
        for c in citations
    ]
    container.conversations.append_message(conversation.id, "user", question)
    container.conversations.append_message(
        conversation.id, "assistant", answer, citations=citation_dicts
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, container: _Container, tenant: Tenant) -> QueryResponse:
    conversation, history_dicts = _resolve_conversation(container, tenant, request)
    try:
        result: AnswerResult = container.answer_service.answer(
            question=request.question,
            top_k=request.top_k,
            metadata_filter=_build_filter(tenant, request.document_ids),
            mode=request.mode,
            conversation_history=history_dicts,
        )
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM provider error: {exc}",
        ) from exc

    if conversation is not None:
        _persist_turn(container, conversation, request.question, result.answer, result.citations)
        conversation = container.conversations.get(conversation.id, tenant)

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                document_id=c.document_id,
                document_name=c.document_name,
                page=c.page,
                chunk_id=c.chunk_id,
                score=c.score,
            )
            for c in (result.citations if request.include_citations else [])
        ],
        retrieval=RetrievalInfo(
            chunks_returned=len(result.chunks),
            latency_ms=result.retrieval_latency_ms,
            mode=result.retrieval_mode,
            reranked=result.reranked,
            chunks=[
                RetrievedChunkOut(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page=chunk.page,
                    score=chunk.score,
                    text=chunk.text,
                )
                for chunk in (result.chunks if request.include_chunks else [])
            ],
        ),
        usage=UsageInfo(
            prompt_tokens=result.usage.get("prompt_tokens", 0),
            completion_tokens=result.usage.get("completion_tokens", 0),
            total_tokens=result.usage.get("total_tokens", 0),
            cost_usd=result.usage.get("cost_usd", 0.0),
            model=result.usage.get("model", ""),
        ),
        insufficient_context=result.insufficient_context,
        groundedness_score=result.groundedness_score,
        prompt_version=result.prompt_version,
        latency_ms=result.total_latency_ms,
        cached=result.cached,
        metadata={"prompt_version": result.prompt_version},
        conversation_id=conversation.id if conversation else None,
        conversation_title=conversation.title if conversation else None,
    )


@router.post("/query/stream")
def query_stream(
    request: QueryRequest, container: _Container, tenant: Tenant
) -> StreamingResponse:
    """Server-sent events: retrieval, delta*, then done (or error)."""

    # Resolve (and validate) the conversation before the stream starts so a
    # bad conversation_id surfaces as a normal 404 response.
    conversation, history_dicts = _resolve_conversation(container, tenant, request)

    def event_stream():
        try:
            for event in container.answer_service.stream_answer(
                question=request.question,
                top_k=request.top_k,
                metadata_filter=_build_filter(tenant, request.document_ids),
                mode=request.mode,
                conversation_history=history_dicts,
            ):
                if conversation is not None and event["type"] == "done":
                    data = event["data"]
                    _persist_turn(
                        container,
                        conversation,
                        request.question,
                        data.get("answer", ""),
                        _stream_citations(data.get("citations", [])),
                    )
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        except LLMError as exc:
            payload = json.dumps({"detail": f"LLM provider error: {exc}"})
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class _StreamCitation:
    """Attribute view over a citation dict from a stream's done event."""

    def __init__(self, data: dict) -> None:
        self.document_id = data.get("document_id", "")
        self.document_name = data.get("document_name", "")
        self.page = data.get("page", 0)
        self.chunk_id = data.get("chunk_id", "")
        self.score = data.get("score", 0.0)


def _stream_citations(raw: list[dict]) -> list[_StreamCitation]:
    return [_StreamCitation(citation) for citation in raw]


@router.post("/retrieval/preview", response_model=RetrievalPreviewResponse)
def preview_retrieval(
    request: RetrievalPreviewRequest, container: _Container, tenant: Tenant
) -> RetrievalPreviewResponse:
    """Preview retrieval results, vector similarity scores, and BM25 scores.

    Runs scoring without LLM generation to allow fast debugging.
    """
    metadata_filter = _build_filter(tenant, request.document_ids)
    raw = container.retriever.debug_retrieve(
        question=request.question,
        top_k=request.top_k,
        metadata_filter=metadata_filter,
        mode=request.mode,
    )

    fused_chunks = [
        RetrievedChunkOut(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page=chunk.page,
            score=chunk.score,
            text=chunk.text,
        )
        for chunk in raw["fused_chunks"]
    ]

    vector_hits = [
        DebugScoreItem(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_name=item["document_name"],
            page=item["page"],
            score=item["score"],
            rank=item["rank"],
            text=item["text"],
        )
        for item in raw["vector_hits"]
    ]

    bm25_hits = [
        DebugScoreItem(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_name=item["document_name"],
            page=item["page"],
            score=item["score"],
            rank=item["rank"],
            text=item["text"],
        )
        for item in raw["bm25_hits"]
    ]

    return RetrievalPreviewResponse(
        question=raw["question"],
        mode=raw["mode"],
        latency_ms=raw["latency_ms"],
        fused_chunks=fused_chunks,
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
        subqueries=raw.get("subqueries", []),
    )

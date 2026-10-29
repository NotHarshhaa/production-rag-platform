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
from app.generation.answer import LLMError

router = APIRouter(tags=["query"])

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _build_filter(tenant: str, document_ids: list[str] | None) -> dict:
    # Tenant scoping is enforced server-side, never taken from the request.
    metadata_filter: dict = {"tenant_id": tenant}
    if document_ids:
        metadata_filter["document_id"] = document_ids
    return metadata_filter


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, container: _Container, tenant: Tenant) -> QueryResponse:
    history_dicts = (
        [m.model_dump() for m in request.conversation_history]
        if request.conversation_history
        else None
    )
    try:
        result = container.answer_service.answer(
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
                for chunk in result.chunks
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
    )


@router.post("/query/stream")
def query_stream(
    request: QueryRequest, container: _Container, tenant: Tenant
) -> StreamingResponse:
    """Server-sent events: retrieval, delta*, then done (or error)."""
    history_dicts = (
        [m.model_dump() for m in request.conversation_history]
        if request.conversation_history
        else None
    )

    def event_stream():
        try:
            for event in container.answer_service.stream_answer(
                question=request.question,
                top_k=request.top_k,
                metadata_filter=_build_filter(tenant, request.document_ids),
                mode=request.mode,
                conversation_history=history_dicts,
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        except LLMError as exc:
            payload = json.dumps({"detail": f"LLM provider error: {exc}"})
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
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
    )


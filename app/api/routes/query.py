"""Grounded question answering: POST /query and POST /query/stream."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import ServiceContainer, get_container
from app.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievalInfo,
    RetrievedChunkOut,
    UsageInfo,
)
from app.generation.answer import LLMError

router = APIRouter(tags=["query"])

_Container = Annotated[ServiceContainer, Depends(get_container)]


def _build_filter(document_ids: list[str] | None) -> dict | None:
    if not document_ids:
        return None
    return {"document_id": document_ids}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, container: _Container) -> QueryResponse:
    try:
        result = container.answer_service.answer(
            question=request.question,
            top_k=request.top_k,
            metadata_filter=_build_filter(request.document_ids),
            mode=request.mode,
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
            model=result.usage.get("model", ""),
        ),
        insufficient_context=result.insufficient_context,
        prompt_version=result.prompt_version,
        latency_ms=result.total_latency_ms,
        metadata={"prompt_version": result.prompt_version},
    )


@router.post("/query/stream")
def query_stream(request: QueryRequest, container: _Container) -> StreamingResponse:
    """Server-sent events: retrieval, delta*, then done (or error)."""

    def event_stream():
        try:
            for event in container.answer_service.stream_answer(
                question=request.question,
                top_k=request.top_k,
                metadata_filter=_build_filter(request.document_ids),
                mode=request.mode,
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
        except LLMError as exc:
            payload = json.dumps({"detail": f"LLM provider error: {exc}"})
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

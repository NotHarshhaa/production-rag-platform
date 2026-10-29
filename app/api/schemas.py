"""Pydantic request/response models. These define the public API contract."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    name: str
    content_type: str
    size_bytes: int
    tenant_id: str = "public"
    status: Literal["uploaded", "indexed", "failed"]
    num_chunks: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    duplicate: bool = False


class DocumentChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page: int
    text: str
    char_count: int
    tenant_id: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_citations: bool = True
    # Restrict retrieval to specific documents (document UUIDs).
    document_ids: list[str] | None = None
    # "hybrid" combines BM25 + vectors; "semantic" uses vectors only.
    mode: Literal["hybrid", "semantic"] | None = None
    # Optional multi-turn conversation context for follow-up questions.
    conversation_history: list[ChatMessage] | None = Field(default=None, max_length=20)


class CitationOut(BaseModel):
    document_id: str
    document_name: str
    page: int
    chunk_id: str
    score: float


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float
    text: str


class RetrievalInfo(BaseModel):
    chunks_returned: int
    latency_ms: float
    mode: str
    reranked: bool = False
    chunks: list[RetrievedChunkOut] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    model: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieval: RetrievalInfo
    usage: UsageInfo
    insufficient_context: bool
    prompt_version: str
    latency_ms: float
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

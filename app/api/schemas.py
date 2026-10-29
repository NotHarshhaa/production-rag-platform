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


class DocumentRawTextOut(BaseModel):
    document_id: str
    name: str
    num_chunks: int
    total_chars: int
    raw_text: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_citations: bool = True
    # Include the retrieved chunk texts in the response (set false to keep
    # responses small when only the answer is consumed).
    include_chunks: bool = True
    # Restrict retrieval to specific documents (document UUIDs).
    document_ids: list[str] | None = None
    # "hybrid" combines BM25 + vectors; "semantic" uses vectors only;
    # "agentic" decomposes the question into sub-queries and fuses results.
    mode: Literal["hybrid", "semantic", "agentic"] | None = None
    # Optional multi-turn conversation context for follow-up questions.
    # Provide conversation_id to use a server-side session instead.
    conversation_history: list[ChatMessage] | None = Field(default=None, max_length=20)
    # Server-side conversation session: history is loaded from storage and
    # this turn (question + answer) is appended to it.
    conversation_id: str | None = None


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
    groundedness_score: float = 1.0
    prompt_version: str
    latency_ms: float
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    conversation_title: str | None = None


class RetrievalPreviewRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: list[str] | None = None
    mode: Literal["hybrid", "semantic", "agentic"] | None = None


class DebugScoreItem(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float
    rank: int
    text: str


class RetrievalPreviewResponse(BaseModel):
    question: str
    mode: str
    latency_ms: float
    fused_chunks: list[RetrievedChunkOut]
    vector_hits: list[DebugScoreItem]
    bm25_hits: list[DebugScoreItem]
    # Populated for mode="agentic": the sub-queries that were fused.
    subqueries: list[str] = Field(default_factory=list)


# --- Conversations ---------------------------------------------------------


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    id: str
    title: str
    tenant_id: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationMessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationOut] = Field(default_factory=list)
    created_at: datetime


class ConversationDetailOut(ConversationOut):
    messages: list[ConversationMessageOut] = Field(default_factory=list)


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


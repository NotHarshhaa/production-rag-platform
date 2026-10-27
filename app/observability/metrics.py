"""Prometheus metrics for HTTP traffic, retrieval, generation, and ingestion.

All metrics live on the default registry and are exported at GET /metrics.
Document content and user questions are never used as metric labels.
"""

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

metrics_router = APIRouter(tags=["metrics"])

# --- HTTP ----------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "rag_http_requests_total",
    "Total HTTP requests processed.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "route"],
)
IN_PROGRESS = Gauge(
    "rag_http_requests_in_progress",
    "HTTP requests currently being processed.",
)

# --- Retrieval -----------------------------------------------------------
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving chunks for a query.",
)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks returned per retrieval.",
    ["mode"],
)

# --- Generation ----------------------------------------------------------
LLM_LATENCY = Histogram(
    "rag_llm_latency_seconds",
    "LLM generation latency in seconds.",
    ["provider"],
)
LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "LLM tokens consumed.",
    ["type"],
)
LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "LLM provider errors.",
    ["provider"],
)

# --- Ingestion -----------------------------------------------------------
INGESTION_LATENCY = Histogram(
    "rag_ingestion_latency_seconds",
    "Document processing latency in seconds.",
    ["operation"],
)
DOCUMENTS_INDEXED = Counter(
    "rag_documents_indexed_total",
    "Documents indexed, by result.",
    ["result"],
)


@metrics_router.get("/metrics")
def expose_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

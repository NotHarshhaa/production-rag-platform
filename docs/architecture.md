# Architecture

This document describes how the platform is put together, the decisions
behind it, and where to extend it.

## High-level flow

```
Client ──> FastAPI ──> Query processing
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Embedding model          BM25 keyword index
             │                       │
             ▼                       ▼
        Vector store ───> Hybrid fusion (RRF)
                         │
                         ▼
                  Optional reranker
                         │
                         ▼
                   Context builder
                         │
                         ▼
                    LLM provider ──> Grounded answer + citations
```

Ingestion runs in parallel: `PDF / TXT / Markdown → text extraction →
chunking → embeddings → vector store`, with document metadata in the
relational database.

## Modules

| Module | Responsibility |
|---|---|
| `app/api` | FastAPI routes, request/response schemas, dependency wiring (`ServiceContainer`) |
| `app/core` | Settings (pydantic-settings) and structured logging with request IDs |
| `app/ingestion` | Loaders (PyMuPDF for PDF), paragraph-aware chunking, the ingest/index pipeline |
| `app/retrieval` | Embedding providers, vector stores, BM25 + hybrid search, reranking |
| `app/generation` | Grounded prompts, citation parsing, the answer service, LLM providers |
| `app/evaluation` | JSONL datasets, retrieval/answer metrics, the evaluation runner |
| `app/observability` | Prometheus metrics and optional OpenTelemetry tracing |
| `app/storage` | SQLAlchemy engine + `documents` table (SQLite or PostgreSQL) |

`app/api/dependencies.py` builds one `ServiceContainer` per application
instance (lifespan-managed). Every component is constructed from `Settings`,
so swapping implementations is a configuration change, not a code change.

## Key design decisions

**Vector store: file-backed locally, Qdrant in production.** `VectorStore`
is an interface with `LocalVectorStore` (NumPy cosine + JSON persistence, the
dev default so CLI workflows share state across processes),
`InMemoryVectorStore` (used by tests), and `QdrantVectorStore` (production,
wired in `compose.yaml`).
Both store chunk text and metadata alongside vectors so retrieval needs no
secondary lookup. A pgvector backend is the natural next adapter.

**Embeddings: hashing by default.** `HashingEmbeddings` (deterministic
feature hashing, no model download) keeps the platform runnable everywhere —
tests, CI, and offline evaluation. For real semantic quality, switch to
`EMBEDDING_PROVIDER=sentence-transformers` (extra `embeddings-local`) or the
`openai` provider. All providers share the `EmbeddingProvider` interface.

**Hybrid retrieval with reciprocal rank fusion.** Vector results and BM25
results are fused with `score = Σ 1/(60 + rank)`. The BM25 index is a pure
Python Okapi implementation rebuilt lazily whenever the vector store's
revision counter changes, so new documents are searchable immediately.

**Single-process caveat.** The revision counter and BM25 index are
process-local. Run the API with one worker (the default) while
`HYBRID_RETRIEVAL` is enabled. A shared index (e.g. PostgreSQL FTS) is the
scaling path.

**Chunking: paragraph-aware, page-bounded.** Chunks never span pages, so
every citation carries an exact page number. Chunk size and overlap are
character-based and configurable (`CHUNK_SIZE`, `CHUNK_OVERLAP`); long
paragraphs fall back to a sliding window with the same overlap.

**Grounded generation.** The system prompt instructs the model to answer
only from numbered passages, cite them as `[n]`, treat document content as
untrusted data, and admit when context is insufficient. `AnswerService`
parses `[n]` markers back into chunk citations; if the model cites nothing,
the top retrieved chunk is attached. With no retrieval results, the service
returns an explicit "not found" answer instead of calling the model.

**LLM providers.** `StubLLM` (deterministic extractive answers) and
`OpenAIChatLLM` (any OpenAI-compatible endpoint, with timeouts and retries)
share the `LLMProvider` interface; `POST /query/stream` uses the same
interface's `stream` method over SSE.

## API surface

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/documents/upload` | Validate + store a document (dedup by SHA-256) |
| POST | `/documents/{id}/index` | Index or re-index (idempotent, replaces old vectors) |
| GET | `/documents` | List documents and their status |
| GET/DELETE | `/documents/{id}` | Fetch or delete a document and its vectors |
| POST | `/query` | Grounded answer + citations + retrieval details |
| POST | `/query/stream` | SSE stream: `retrieval`, `delta*`, `done` |
| GET | `/health` | Component checks (database, vector store) |
| GET | `/metrics` | Prometheus metrics |

## Metrics

- `rag_http_requests_total{method,route,status}`, `..._latency_seconds`, `rag_http_requests_in_progress`
- `rag_retrieval_latency_seconds`, `rag_retrieved_chunks{mode}`
- `rag_llm_latency_seconds{provider}`, `rag_llm_tokens_total{type}`, `rag_llm_errors_total`
- `rag_ingestion_latency_seconds{operation}`, `rag_documents_indexed_total{result}`

A provisioned Grafana dashboard (`RAG Platform Overview`) ships in
`docker/grafana/dashboards/`.

## Security notes

Uploads are extension-validated and size-capped before the body is read.
Document text is never logged; logs carry ids, statuses, and timings. The
prompt treats retrieved content as data, and per-document filters enforce
tenant isolation at retrieval time. These are the MVP controls — the README
checklist (auth, retention, per-request authorization) remains open work.

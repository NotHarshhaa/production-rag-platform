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
| `app/api` | FastAPI routes (documents, query, conversations, maintenance, health), request/response schemas, API-key auth with per-key quotas (`auth.py`), dependency wiring (`ServiceContainer`) |
| `app/core` | Settings (pydantic-settings), structured logging with request IDs, shared sliding-window rate limiting |
| `app/ingestion` | Loaders (PyMuPDF for PDF), paragraph-aware chunking, the ingest/index pipeline, background workers (`worker.py`), retention sweeps (`retention.py`) |
| `app/retrieval` | Embedding providers, vector stores, BM25 + hybrid search, agentic sub-query decomposition (`query_decomposer.py`), reranking |
| `app/generation` | Grounded prompts, citation parsing, the answer service with query cache and cost tracking, LLM providers |
| `app/evaluation` | JSONL datasets, retrieval/answer metrics, LLM-as-judge grading (`judge.py`), the evaluation runner |
| `app/observability` | Prometheus metrics and optional OpenTelemetry tracing |
| `app/storage` | SQLAlchemy engine + tenant-scoped `documents` and `conversations` tables (SQLite or PostgreSQL) |

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
revision counter changes, so new documents are searchable immediately. The
same rebuild refreshes a chunk-key → metadata map, so BM25 hits outside the
vector top-N resolve without re-scanning the store.

**Agentic retrieval (mode="agentic").** Compound questions are decomposed
into focused sub-queries (`query_decomposer.py`): an LLM-based splitter when
a real provider is configured, a sentence/conjunction heuristic otherwise
(with the stub provider). Each sub-query is retrieved independently and the
result lists are fused with the same RRF formula, so agentic mode degrades
gracefully to plain retrieval for simple questions — the original question
is always one of the sub-queries.

**Server-side conversations.** `/conversations` stores chat threads per
tenant (`conversations` + `conversation_messages` tables). `/query` accepts
`conversation_id`: the stored turns are replayed into the prompt (bounded by
`CONVERSATION_MAX_TURNS`) and the new question/answer pair is appended, so
clients never need to resend history. Citation objects are persisted with
assistant turns for faithful re-rendering.

**Retention.** With `RETENTION_DAYS > 0`, a daemon sweeper
(`app/ingestion/retention.py`) deletes documents whose `updated_at` is older
than the window; re-indexing or replacing a document resets its age. The
same sweep runs on demand via the admin-gated `POST /maintenance/retention`.

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

**Multi-tenancy via API keys.** When `AUTH_ENABLED=true`, every request
must present an API key (`X-API-Key` or `Authorization: Bearer`); keys map
to tenants as `key:tenant` pairs. `documents.tenant_id` scopes rows, and
retrieval filters by `tenant_id` server-side, so one tenant's documents can
never surface in another tenant's results. With auth disabled everything
runs in the default `public` tenant. `/health` and `/metrics` stay open for
probes and scraping.

**Background ingestion.** With `INGESTION_MODE=background`, index requests
submit to a small `ThreadPoolExecutor` (`app/ingestion/worker.py`) and
return 202 immediately; documents move `uploaded -> indexed/failed` and
clients poll `GET /documents/{id}`. The thread pool shares the process's
vector store, so the single-worker caveat below applies equally.

**Query cache with corpus invalidation.** `AnswerService` caches answers
per (tenant, question, top_k, filter, mode) with a TTL. Each entry records
the vector store revision at insert time; any document index or delete
bumps that revision and invalidates the cache, so answers never go stale
after corpus changes. Streaming responses bypass the cache.

**Cost tracking.** Configure `LLM_COST_PROMPT_PER_1K` /
`LLM_COST_COMPLETION_PER_1K` with your provider's prices; every answer's
estimated cost is included in the response (`usage.cost_usd`) and aggregated
in the `rag_llm_cost_usd_total` metric.

**Document update workflow.** `PUT /documents/{id}` replaces a document's
content (rejecting hash conflicts with 409), resets its state, and re-indexes
— idempotent like the index endpoint.

**LLM providers.** `StubLLM` (deterministic extractive answers) and
`OpenAIChatLLM` (any OpenAI-compatible endpoint, with timeouts and retries)
share the `LLMProvider` interface; `POST /query/stream` uses the same
interface's `stream` method over SSE.

## API surface

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/documents/upload` | Validate + store a document (per-tenant dedup by SHA-256; `?auto_index=true` to index immediately) |
| POST | `/documents/{id}/index` | Index or re-index (idempotent; 202 when background ingestion is on) |
| PUT | `/documents/{id}` | Replace a document's content and re-index |
| GET | `/documents` | List documents and their status (own tenant only) |
| GET/DELETE | `/documents/{id}` | Fetch or delete a document and its vectors |
| POST | `/query` | Grounded answer + citations + retrieval details (cache-aware; `conversation_id` continues a stored thread) |
| POST | `/query/stream` | SSE stream: `retrieval`, `delta*`, `done` |
| POST | `/retrieval/preview` | Retrieval debugging: fused/vector/BM25 scores, agentic sub-queries |
| POST/GET | `/conversations` | Start / list server-side chat sessions |
| GET/PATCH/DELETE | `/conversations/{id}` | Fetch with messages / rename / delete |
| POST | `/maintenance/retention` | Run a retention sweep now (`ADMIN_KEYS` required when auth is on) |
| GET | `/health` | Component checks + document/chunk stats — unauthenticated |
| GET | `/metrics` | Prometheus metrics — unauthenticated |

All `/documents`, `/query`, and `/conversations` routes require an API key
when `AUTH_ENABLED=true`; see `app/api/auth.py` for the header formats and
the optional `key:tenant:quota` per-key rate limits.

## Metrics

- `rag_http_requests_total{method,route,status}`, `..._latency_seconds`, `rag_http_requests_in_progress`
- `rag_retrieval_latency_seconds`, `rag_retrieved_chunks{mode}`
- `rag_llm_latency_seconds{provider}`, `rag_llm_tokens_total{type}`, `rag_llm_errors_total`
- `rag_llm_cost_usd_total` (estimated spend), `rag_query_cache_total{result}` (hit/miss)
- `rag_ingestion_latency_seconds{operation}`, `rag_documents_indexed_total{result}`
- `rag_auth_quota_rejections_total` (per-API-key quota 429s)

A provisioned Grafana dashboard (`RAG Platform Overview`) ships in
`docker/grafana/dashboards/`.

## Security notes

Uploads are extension-validated and size-capped before the body is read.
Document text is never logged; logs carry ids, statuses, and timings. The
prompt treats retrieved content as data. Tenant isolation is enforced
server-side at retrieval time via API-key-mapped tenants. Per-key request
quotas return 429 with `Retry-After`, and maintenance endpoints require a
key listed in `ADMIN_KEYS` when auth is enabled. Still open: full RBAC
beyond tenant scoping and admin keys.

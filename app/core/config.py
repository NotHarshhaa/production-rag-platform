"""Application settings, loaded from environment variables and a local .env file.

The environment variable names follow the contract documented in README.md
(APP_ENV, DATABASE_URL, VECTOR_STORE, ...) and are matched case-insensitively
by pydantic-settings.
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VECTOR_STORE_LOCAL = "local"
VECTOR_STORE_INMEMORY = "inmemory"
VECTOR_STORE_QDRANT = "qdrant"

EMBEDDING_HASHING = "hashing"
EMBEDDING_SENTENCE_TRANSFORMERS = "sentence-transformers"
EMBEDDING_OPENAI = "openai"

LLM_STUB = "stub"
LLM_OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: str = "development"
    app_name: str = "production-rag-platform"
    version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    # Storage
    database_url: str = "sqlite:///./data/rag.db"

    # Vector store: local (file-backed, dev default), inmemory (tests), qdrant.
    vector_store: str = VECTOR_STORE_LOCAL
    vector_store_path: str = "./data/vectors.json"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_chunks"

    # Embeddings
    embedding_provider: str = EMBEDDING_HASHING
    embedding_model: str = ""
    embedding_dimension: int = 384

    # Chunking (character-based)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20
    hybrid_retrieval: bool = True
    min_relevance_score: float = 0.0

    # Reranking
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    llm_provider: str = LLM_STUB
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Cost estimation (currency units per 1k tokens; 0 disables tracking)
    llm_cost_prompt_per_1k: float = 0.0
    llm_cost_completion_per_1k: float = 0.0

    # Uploads
    max_upload_size_mb: int = 25
    allowed_extensions: str = "pdf,txt,md"

    # Authentication and multi-tenancy
    auth_enabled: bool = False
    # Comma-separated "key:tenant" pairs; a bare key maps to default_tenant.
    # An optional third segment sets a per-key request quota per minute
    # ("key:tenant:60"); 0 or omission means unlimited.
    api_keys: str = ""
    default_tenant: str = "public"
    # Comma-separated API keys allowed to call maintenance endpoints
    # (e.g. POST /maintenance/retention). Ignored when auth is disabled.
    admin_keys: str = ""

    # Ingestion: "sync" (blocking) or "background" (thread-pool workers)
    ingestion_mode: str = "sync"
    ingestion_workers: int = 2

    # Retention: delete documents untouched for this many days (0 disables).
    # Sweeps run in a background thread and on demand via
    # POST /maintenance/retention.
    retention_days: int = 0
    retention_sweep_interval_seconds: int = 3600

    # Conversations: server-side chat sessions. At most this many past turns
    # are replayed into the prompt (0 keeps every stored turn).
    conversation_max_turns: int = 10

    # Agentic retrieval: split the question into at most this many sub-queries
    # and fuse their results (used when mode="agentic").
    agentic_max_subqueries: int = 3

    # Query cache (seconds; 0 disables). Entries are invalidated whenever
    # the corpus changes (any document indexed or deleted).
    query_cache_ttl_seconds: int = 300

    # Observability
    log_level: str = "INFO"
    log_json: bool = False
    metrics_enabled: bool = True
    otel_enabled: bool = False
    otel_endpoint: str | None = None

    # Rate limiting (requests per minute per client IP; 0 disables)
    rate_limit_per_minute: int = 0

    @model_validator(mode="after")
    def _validate_ranges(self) -> "Settings":
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters")
        self.chunk_overlap = max(0, min(self.chunk_overlap, self.chunk_size - 1))
        if self.retrieval_top_k < 1 or self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError(
                "RETRIEVAL_TOP_K must be >= 1 and <= RETRIEVAL_CANDIDATE_K"
            )
        if self.max_upload_size_mb < 1:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be at least 1")
        if self.auth_enabled and not self.api_keys.strip():
            raise ValueError("AUTH_ENABLED=true requires at least one entry in API_KEYS")
        if self.ingestion_mode not in ("sync", "background"):
            raise ValueError("INGESTION_MODE must be 'sync' or 'background'")
        if self.ingestion_workers < 1:
            raise ValueError("INGESTION_WORKERS must be at least 1")
        if self.retention_days < 0:
            raise ValueError("RETENTION_DAYS must be 0 (disabled) or positive")
        if self.retention_sweep_interval_seconds < 60:
            raise ValueError("RETENTION_SWEEP_INTERVAL_SECONDS must be at least 60")
        if self.agentic_max_subqueries < 1:
            raise ValueError("AGENTIC_MAX_SUBQUERIES must be at least 1")
        self.log_level = self.log_level.upper()
        return self

    @property
    def allowed_extension_list(self) -> list[str]:
        return [
            f".{ext.strip().lstrip('.').lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        ]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor for scripts and the default app factory."""
    return Settings()

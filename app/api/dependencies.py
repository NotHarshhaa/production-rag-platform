"""Wiring of the service container: one graph of components per app instance."""

from dataclasses import dataclass, field

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.ratelimit import SlidingWindowLimiter
from app.generation.answer import AnswerService, build_llm_provider
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.retention import RetentionSweeper
from app.ingestion.worker import BackgroundIndexer
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.retrieval.vector_store import build_vector_store
from app.storage.database import build_engine, init_db
from app.storage.repository import ConversationRepository, DocumentRepository


@dataclass
class ServiceContainer:
    """All runtime components, built once per application instance."""

    settings: Settings
    repository: DocumentRepository
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    reranker: object
    answer_service: AnswerService
    conversations: ConversationRepository
    indexer: BackgroundIndexer | None = None
    retention_sweeper: RetentionSweeper | None = None
    # Per-key quota limiters, keyed by the quota value (requests per minute).
    key_limiters: dict[int, SlidingWindowLimiter] = field(default_factory=dict)

    @property
    def background_ingestion(self) -> bool:
        return self.indexer is not None

    def shutdown(self) -> None:
        if self.indexer is not None:
            self.indexer.shutdown()


def build_container(settings: Settings | None = None) -> ServiceContainer:
    settings = settings or get_settings()

    engine = build_engine(settings.database_url)
    init_db(engine)
    repository = DocumentRepository(engine)
    conversations = ConversationRepository(engine)

    embeddings = build_embedding_provider(settings)
    # Prime the provider so the vector store is created with the right dimension.
    probe = embeddings.embed_query("dimension probe")
    dimension = len(probe)
    vector_store = build_vector_store(settings, dimension)

    llm = build_llm_provider(settings)
    retriever = HybridRetriever(embeddings, vector_store, settings, llm=llm)
    reranker = build_reranker(settings)
    pipeline = IngestionPipeline(repository, embeddings, vector_store, settings)
    answer_service = AnswerService(retriever, reranker, llm, settings)
    indexer = (
        BackgroundIndexer(pipeline, settings.ingestion_workers)
        if settings.ingestion_mode == "background"
        else None
    )
    retention_sweeper = (
        RetentionSweeper(
            pipeline,
            retention_days=settings.retention_days,
            interval_seconds=settings.retention_sweep_interval_seconds,
        )
        if settings.retention_days > 0
        else None
    )

    return ServiceContainer(
        settings=settings,
        repository=repository,
        pipeline=pipeline,
        retriever=retriever,
        reranker=reranker,
        answer_service=answer_service,
        conversations=conversations,
        indexer=indexer,
        retention_sweeper=retention_sweeper,
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container

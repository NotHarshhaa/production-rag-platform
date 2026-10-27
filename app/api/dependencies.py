"""Wiring of the service container: one graph of components per app instance."""

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings, get_settings
from app.generation.answer import AnswerService, build_llm_provider
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.worker import BackgroundIndexer
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.retrieval.vector_store import build_vector_store
from app.storage.database import build_engine, init_db
from app.storage.repository import DocumentRepository


@dataclass
class ServiceContainer:
    """All runtime components, built once per application instance."""

    settings: Settings
    repository: DocumentRepository
    pipeline: IngestionPipeline
    retriever: HybridRetriever
    reranker: object
    answer_service: AnswerService
    indexer: BackgroundIndexer | None = None

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

    embeddings = build_embedding_provider(settings)
    # Prime the provider so the vector store is created with the right dimension.
    probe = embeddings.embed_query("dimension probe")
    dimension = len(probe)
    vector_store = build_vector_store(settings, dimension)

    retriever = HybridRetriever(embeddings, vector_store, settings)
    reranker = build_reranker(settings)
    llm = build_llm_provider(settings)
    pipeline = IngestionPipeline(repository, embeddings, vector_store, settings)
    answer_service = AnswerService(retriever, reranker, llm, settings)
    indexer = (
        BackgroundIndexer(pipeline, settings.ingestion_workers)
        if settings.ingestion_mode == "background"
        else None
    )

    return ServiceContainer(
        settings=settings,
        repository=repository,
        pipeline=pipeline,
        retriever=retriever,
        reranker=reranker,
        answer_service=answer_service,
        indexer=indexer,
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container

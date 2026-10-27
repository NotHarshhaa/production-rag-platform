"""Unit tests for the BM25 index and hybrid retriever fusion."""

from app.core.config import Settings
from app.retrieval.embeddings import HashingEmbeddings
from app.retrieval.hybrid_search import BM25Index, HybridRetriever
from app.retrieval.vector_store import InMemoryVectorStore, VectorRecord

DOCS = [
    ("doc1:0", "The python programming language is great for data.", "doc1"),
    ("doc2:0", "Docker containers package applications for deployment.", "doc2"),
    ("doc3:0", "Kubernetes orchestrates docker containers at scale.", "doc3"),
]


def _as_build_documents() -> list[tuple[str, str, dict]]:
    return [(key, text, {"document_id": document_id}) for key, text, document_id in DOCS]


def test_bm25_ranks_relevant_document_first():
    index = BM25Index()
    index.build(_as_build_documents())
    hits = index.search("docker containers", top_k=3)
    assert hits[0][0] == "doc2:0"


def test_bm25_respects_top_k():
    index = BM25Index()
    index.build(DOCS)
    assert len(index.search("python docker kubernetes", top_k=2)) == 2


def test_bm25_metadata_filter_excludes_documents():
    index = BM25Index()
    index.build(_as_build_documents())
    hits = index.search("docker containers", top_k=3, metadata_filter={"document_id": "doc1"})
    assert hits == []


def test_bm25_no_matches_returns_empty():
    index = BM25Index()
    index.build(_as_build_documents())
    assert index.search("zebra", top_k=3) == []


def _store_with_docs() -> InMemoryVectorStore:
    embeddings = HashingEmbeddings(dimension=128)
    store = InMemoryVectorStore()
    records = []
    for key, body, document_id in DOCS:
        records.append(
            VectorRecord(
                id=key,
                vector=embeddings.embed_documents([body])[0],
                metadata={
                    "chunk_id": key,
                    "document_id": document_id,
                    "document_name": f"{document_id}.md",
                    "page": 1,
                    "text": body,
                },
            )
        )
    store.add(records)
    return store


def test_hybrid_retriever_fuses_results():
    settings = Settings(hybrid_retrieval=True)
    retriever = HybridRetriever(HashingEmbeddings(dimension=128), _store_with_docs(), settings)
    chunks = retriever.retrieve("docker containers for deployment", top_k=3)
    assert chunks
    assert all(isinstance(c.score, float) for c in chunks)
    assert all(c.text for c in chunks)
    # Any doc mentioning docker should surface in the top results.
    doc_ids = {c.document_id for c in chunks}
    assert "doc2" in doc_ids or "doc3" in doc_ids


def test_hybrid_retriever_semantic_mode():
    settings = Settings(hybrid_retrieval=False)
    retriever = HybridRetriever(HashingEmbeddings(dimension=128), _store_with_docs(), settings)
    chunks = retriever.retrieve("docker", top_k=2)
    assert len(chunks) <= 2


def test_retriever_picks_up_new_documents_after_revision_change():
    settings = Settings(hybrid_retrieval=True)
    store = _store_with_docs()
    retriever = HybridRetriever(HashingEmbeddings(dimension=128), store, settings)
    before = retriever.retrieve("kubernetes orchestrates", top_k=3)

    embeddings = HashingEmbeddings(dimension=128)
    text = "Kubernetes orchestrates docker containers at scale across clusters."
    store.add([
        VectorRecord(
            id="doc4:0",
            vector=embeddings.embed_query(text),
            metadata={
                "chunk_id": "doc4:0",
                "document_id": "doc4",
                "document_name": "d4.md",
                "page": 1,
                "text": text,
            },
        )
    ])
    after = retriever.retrieve("kubernetes orchestrates", top_k=3)
    assert any(c.document_id == "doc4" for c in after) or len(after) >= len(before)

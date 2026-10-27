"""Unit tests for the in-memory vector store."""

from app.retrieval.vector_store import InMemoryVectorStore, VectorRecord, matches_filter


def _record(record_id: str, document_id: str, vector: list[float]) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata={
            "chunk_id": record_id,
            "document_id": document_id,
            "document_name": f"{document_id}.md",
            "page": 1,
            "text": f"text of {record_id}",
        },
    )


def test_add_and_search_ranking():
    store = InMemoryVectorStore()
    store.add([
        _record("a:0", "a", [1.0, 0.0, 0.0]),
        _record("b:0", "b", [0.0, 1.0, 0.0]),
    ])
    results = store.search([1.0, 0.05, 0.0], top_k=2)
    assert results[0].id == "a:0"
    assert results[0].score > results[1].score
    assert results[0].metadata["document_id"] == "a"


def test_search_respects_top_k():
    store = InMemoryVectorStore()
    store.add([_record(f"d{i}:0", f"d{i}", [float(i), 1.0, 0.0]) for i in range(5)])
    assert len(store.search([1.0, 1.0, 0.0], top_k=3)) == 3


def test_metadata_filter_any_of():
    store = InMemoryVectorStore()
    store.add([_record("a:0", "a", [1.0, 0.0]), _record("b:0", "b", [0.9, 0.1])])
    results = store.search([1.0, 0.0], top_k=5, metadata_filter={"document_id": ["b"]})
    assert [r.id for r in results] == ["b:0"]


def test_delete_document_removes_only_that_document():
    store = InMemoryVectorStore()
    store.add([_record("a:0", "a", [1.0, 0.0]), _record("b:0", "b", [0.5, 0.5])])
    store.delete_document("a")
    assert store.count_chunks() == 1
    results = store.search([1.0, 0.0], top_k=5)
    assert [r.id for r in results] == ["b:0"]
    assert list(store.iter_chunks())[0].metadata["document_id"] == "b"


def test_revision_increments_on_mutation():
    store = InMemoryVectorStore()
    start = store.get_revision()
    store.add([_record("a:0", "a", [1.0])])
    store.delete_document("a")
    assert store.get_revision() >= start + 2


def test_matches_filter_semantics():
    metadata = {"document_id": "a", "page": 3}
    assert matches_filter(metadata, None)
    assert matches_filter(metadata, {"document_id": "a"})
    assert matches_filter(metadata, {"document_id": ["a", "b"]})
    assert not matches_filter(metadata, {"document_id": "b"})
    assert not matches_filter(metadata, {"missing": "x"})


def test_local_vector_store_persists_across_instances(tmp_path):
    from app.retrieval.vector_store import LocalVectorStore

    path = tmp_path / "vectors.json"
    store = LocalVectorStore(path)
    store.add([_record("a:0", "a", [1.0, 0.0])])
    store.add([_record("b:0", "b", [0.0, 1.0])])
    store.delete_document("b")

    reloaded = LocalVectorStore(path)
    assert reloaded.count_chunks() == 1
    results = reloaded.search([1.0, 0.0], top_k=5)
    assert [r.id for r in results] == ["a:0"]


def test_local_vector_store_survives_corrupt_file(tmp_path):
    from app.retrieval.vector_store import LocalVectorStore

    path = tmp_path / "vectors.json"
    path.write_text("{not json", encoding="utf-8")
    store = LocalVectorStore(path)
    assert store.count_chunks() == 0

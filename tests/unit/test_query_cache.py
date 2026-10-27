"""Unit tests for the query cache."""

from app.generation.answer import QueryCache


def test_put_and_get_roundtrip():
    cache = QueryCache(ttl_seconds=60)
    cache.put("k", {"answer": "x"}, revision=1)
    assert cache.get("k", current_revision=1) == {"answer": "x"}


def test_disabled_cache_never_stores():
    cache = QueryCache(ttl_seconds=0)
    assert cache.enabled is False
    cache.put("k", "value", revision=1)
    assert cache.get("k", current_revision=1) is None


def test_revision_change_invalidates_entries():
    cache = QueryCache(ttl_seconds=60)
    cache.put("k", "value", revision=1)
    assert cache.get("k", current_revision=2) is None
    assert cache.get("k", current_revision=1) is None  # entry was dropped


def test_ttl_expiry(monkeypatch):
    import app.generation.answer as answer_module

    cache = QueryCache(ttl_seconds=10)
    cache.put("k", "value", revision=1)
    assert cache.get("k", current_revision=1) == "value"

    real_monotonic = answer_module.time.monotonic
    monkeypatch.setattr(answer_module.time, "monotonic", lambda: real_monotonic() + 20)
    assert cache.get("k", current_revision=1) is None


def test_max_entries_evicts_oldest():
    cache = QueryCache(ttl_seconds=60, max_entries=2)
    for i in range(3):
        cache.put(f"k{i}", f"value{i}", revision=1)
    assert cache.get("k0", current_revision=1) is None
    assert cache.get("k1", current_revision=1) == "value1"
    assert cache.get("k2", current_revision=1) == "value2"


def test_cache_key_is_deterministic_and_order_insensitive():
    first = QueryCache.make_key("What is RAG?", 5, {"document_id": ["a", "b"]}, "hybrid")
    second = QueryCache.make_key("what is rag?  ", 5, {"document_id": ["a", "b"]}, "hybrid")
    assert first == second
    other = QueryCache.make_key("What is RAG?", 3, {"document_id": ["a", "b"]}, "hybrid")
    assert first != other

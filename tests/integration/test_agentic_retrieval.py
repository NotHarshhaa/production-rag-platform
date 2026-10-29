"""Agentic retrieval mode: query decomposition and multi-query fusion."""

from app.retrieval.query_decomposer import decompose_question


class RecordingLLM:
    """Minimal LLM double that answers decomposition prompts."""

    name = "openai"
    model = "test-llm"

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, messages, temperature, max_tokens):
        from app.generation.answer import LLMResponse

        self.prompts.append(messages[-1]["content"])
        return LLMResponse(text=self.text, model=self.model)


def test_decompose_keeps_original_question_first():
    queries = decompose_question("How does deployment work?", llm=None, max_subqueries=3)
    assert queries[0] == "How does deployment work?"


def test_heuristic_decompose_splits_compound_questions():
    queries = decompose_question(
        "How does deployment work and also how does rollback work?",
        llm=None,
        max_subqueries=3,
    )
    assert queries[0].startswith("How does deployment work")
    assert any("rollback" in q for q in queries[1:])


def test_decompose_never_exceeds_max_subqueries():
    question = (
        "How does deployment work and also how does rollback work; "
        "also how does monitoring work?"
    )
    queries = decompose_question(question, llm=None, max_subqueries=2)
    assert len(queries) == 2


def test_llm_decomposition_is_used_when_available():
    llm = RecordingLLM("- How does deployment work?\n- How does rollback work?\n")
    queries = decompose_question("How do deployment and rollback work?", llm=llm, max_subqueries=3)
    assert len(llm.prompts) == 1
    assert "How does rollback work?" in queries
    # The original question is still first.
    assert queries[0] == "How do deployment and rollback work?"


def test_llm_failure_falls_back_to_heuristics():
    class BrokenLLM(RecordingLLM):
        def generate(self, messages, temperature, max_tokens):
            raise RuntimeError("provider down")

    queries = decompose_question(
        "How does deployment work and also how does rollback work?",
        llm=BrokenLLM(""),
        max_subqueries=3,
    )
    assert any("rollback" in q for q in queries)


def test_agentic_query_mode_end_to_end(seeded_client):
    response = seeded_client.post(
        "/query",
        json={
            "question": (
                "How does the deployment pipeline validate changes and also "
                "how are credentials stored?"
            ),
            "mode": "agentic",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieval"]["mode"] == "agentic"
    assert body["retrieval"]["chunks_returned"] > 0
    assert body["answer"], "agentic mode must still produce a grounded answer"


def test_agentic_preview_returns_subqueries(seeded_client):
    response = seeded_client.post(
        "/retrieval/preview",
        json={
            "question": (
                "How does the deployment pipeline work and also how does "
                "tenant isolation work?"
            ),
            "mode": "agentic",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "agentic"
    assert len(body["subqueries"]) >= 2
    assert body["fused_chunks"], "fused chunks should combine sub-query hits"


def test_agentic_respects_document_selection(seeded_client):
    documents = seeded_client.get("/documents").json()
    deployment_doc = next(d for d in documents if d["name"] == "deployment.md")
    response = seeded_client.post(
        "/query",
        json={
            "question": "How does deployment work and also how are credentials stored?",
            "mode": "agentic",
            "document_ids": [deployment_doc["id"]],
        },
    )
    assert response.status_code == 200
    names = {c["document_name"] for c in response.json()["citations"]}
    assert names <= {"deployment.md"}

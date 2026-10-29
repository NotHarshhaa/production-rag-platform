"""Server-side conversation sessions: /conversations API and /query integration."""

from tests.integration.conftest import DOC_ONE, make_settings


def _seed_corpus(client) -> str:
    response = client.post(
        "/documents/upload",
        files={"file": ("deployment.md", DOC_ONE.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 201, response.text
    document_id = response.json()["id"]
    index_response = client.post(f"/documents/{document_id}/index")
    assert index_response.status_code == 200, index_response.text
    return document_id


def test_create_list_get_delete_conversation(client):
    created = client.post("/conversations", json={"title": "Ops questions"})
    assert created.status_code == 201, created.text
    conversation = created.json()
    assert conversation["title"] == "Ops questions"
    assert conversation["message_count"] == 0

    listing = client.get("/conversations").json()
    assert [c["id"] for c in listing] == [conversation["id"]]

    detail = client.get(f"/conversations/{conversation['id']}")
    assert detail.status_code == 200
    assert detail.json()["messages"] == []

    renamed = client.patch(f"/conversations/{conversation['id']}", json={"title": "Renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"

    deleted = client.delete(f"/conversations/{conversation['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/conversations/{conversation['id']}").status_code == 404
    assert client.get("/conversations").json() == []


def test_missing_conversation_returns_404(client):
    assert client.get("/conversations/does-not-exist").status_code == 404
    assert client.patch(
        "/conversations/does-not-exist", json={"title": "x"}
    ).status_code == 404
    assert client.delete("/conversations/does-not-exist").status_code == 404


def test_query_with_conversation_id_persists_turns(client):
    _seed_corpus(client)
    conversation_id = client.post("/conversations", json={}).json()["id"]

    first = client.post(
        "/query",
        json={
            "question": "How does the deployment pipeline validate changes?",
            "conversation_id": conversation_id,
        },
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["conversation_id"] == conversation_id
    # The title follows the first user question.
    assert "deployment pipeline" in body["conversation_title"].lower()

    second = client.post(
        "/query",
        json={
            "question": "What happens after the build stage produces a Docker image?",
            "conversation_id": conversation_id,
        },
    ).json()

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert detail["message_count"] == 4
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][0]["content"] == "How does the deployment pipeline validate changes?"
    assert detail["messages"][1]["content"] == first.json()["answer"]
    assert detail["messages"][3]["content"] == second["answer"]
    assert (
        detail["messages"][2]["content"]
        == "What happens after the build stage produces a Docker image?"
    )
    # Assistant turns carry resolved citations for re-rendering.
    assert detail["messages"][1]["citations"], "assistant message should store citations"
    assert detail["messages"][1]["citations"][0]["document_name"] == "deployment.md"


def test_conversation_history_is_replayed_into_retrieval(client):
    """The stored history changes the contextualized search query."""

    _seed_corpus(client)
    conversation_id = client.post("/conversations", json={}).json()["id"]
    client.post(
        "/query",
        json={
            "question": "How does the deployment pipeline validate changes?",
            "conversation_id": conversation_id,
        },
    )

    # A pronoun-only follow-up must be expanded with the previous turn.
    from app.generation.answer import _contextual_search_query

    response = client.post(
        "/query",
        json={"question": "Tell me more about it", "conversation_id": conversation_id},
    )
    assert response.status_code == 200
    expanded = _contextual_search_query(
        "Tell me more about it",
        [{"role": "user", "content": "How does the deployment pipeline validate changes?"}],
    )
    assert "deployment" in expanded


def test_unknown_conversation_id_on_query_returns_404(client):
    _seed_corpus(client)
    response = client.post(
        "/query", json={"question": "anything", "conversation_id": "missing"}
    )
    assert response.status_code == 404


def test_stream_query_persists_conversation_turn(client):
    _seed_corpus(client)
    conversation_id = client.post("/conversations", json={}).json()["id"]

    with client.stream(
        "POST",
        "/query/stream",
        json={
            "question": "How does the deployment pipeline work?",
            "conversation_id": conversation_id,
        },
    ) as response:
        assert response.status_code == 200
        events = [
            line.removeprefix("event: ").strip()
            for line in response.iter_lines()
            if line.startswith("event:")
        ]
    assert "done" in events

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert detail["message_count"] == 2
    assert detail["messages"][1]["role"] == "assistant"


def test_conversations_are_tenant_scoped(tmp_path):
    settings = make_settings(
        tmp_path, auth_enabled=True, api_keys="sk-acme:acme,sk-globex:globex"
    )

    from fastapi.testclient import TestClient

    from app.main import create_app as build_app

    with TestClient(build_app(settings)) as client:
        created = client.post(
            "/conversations", json={"title": "private"}, headers={"X-API-Key": "sk-acme"}
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        # The other tenant can neither see nor delete it.
        assert (
            client.get(
                f"/conversations/{conversation_id}", headers={"X-API-Key": "sk-globex"}
            ).status_code
            == 404
        )
        assert client.get("/conversations", headers={"X-API-Key": "sk-globex"}).json() == []
        assert (
            client.delete(
                f"/conversations/{conversation_id}", headers={"X-API-Key": "sk-globex"}
            ).status_code
            == 404
        )
        assert client.get("/conversations", headers={"X-API-Key": "sk-acme"}).json()[0][
            "tenant_id"
        ] == "acme"

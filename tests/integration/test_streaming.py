"""Integration tests for the streaming query endpoint."""

import json


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if event_type:
            events.append((event_type, data))
    return events


def test_stream_emits_retrieval_deltas_and_done(seeded_client):
    with seeded_client.stream(
        "POST", "/query/stream", json={"question": "How does deployment work?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in response.iter_text())

    events = _parse_sse(body)
    types = [event_type for event_type, _ in events]
    assert types[0] == "retrieval"
    assert "delta" in types
    assert types[-1] == "done"

    retrieval_event = events[0][1]
    assert retrieval_event["chunks_returned"] > 0

    deltas = [data["text"] for event_type, data in events if event_type == "delta"]
    assert "".join(deltas).strip()

    done_event = events[-1][1]
    assert done_event["answer"]
    assert "latency_ms" in done_event


def test_stream_with_no_matching_documents(seeded_client):
    with seeded_client.stream(
        "POST",
        "/query/stream",
        json={"question": "How does deployment work?", "document_ids": ["missing"]},
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    events = _parse_sse(body)
    assert [event_type for event_type, _ in events] == ["retrieval", "done"]
    assert events[-1][1]["insufficient_context"] is True

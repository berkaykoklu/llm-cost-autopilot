import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from autopilot.prices import cost
from autopilot.proxy import create_app
from autopilot.store import Store
from offline.train import fit

EASY = "what colour is the sky"
HARD = "prove the theorem about eigenvalues"
URL = "/v1/chat/completions"


def completion(model: str) -> dict[str, Any]:
    return {
        "id": "x", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


class Upstream:
    def __init__(self, status: int = 200, down: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.status, self.down = status, down

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down", request=request)
        body = json.loads(request.content)
        self.calls.append({"body": body, "auth": request.headers["authorization"]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=completion(body["model"]))


@pytest.fixture(scope="module")
def pipe() -> Any:
    texts = [EASY, "what colour is grass", HARD, "prove the lemma about eigenvalues"] * 5
    return fit(texts, [1, 1, 0, 0] * 5)


def make(pipe: Any, upstream: Upstream) -> tuple[TestClient, Store]:
    store = Store(":memory:")
    app = create_app(store, pipe, 0.5, transport=httpx.MockTransport(upstream))
    return TestClient(app), store


def ask(client: TestClient, model: str = "auto", text: str = EASY, key: str = "sk-test-1",
        **extra: Any) -> Any:  # TestClient returns its own httpx-compatible Response type
    body = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})


def test_missing_key_rejected_before_upstream(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    r = client.post(URL, json={"model": "auto", "messages": []})
    assert r.status_code == 401
    assert up.calls == []


def test_stream_rejected(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    assert ask(client, stream=True).status_code == 400
    assert up.calls == []


def test_invalid_body_rejected(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    auth = {"Authorization": "Bearer sk-test-1"}
    assert client.post(URL, content=b"not json", headers=auth).status_code == 400
    assert client.post(URL, json=[1, 2], headers=auth).status_code == 400
    assert up.calls == []


def test_explicit_model_passes_through(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    r = ask(client, model="gpt-4o")
    assert r.status_code == 200
    assert up.calls[0]["body"]["model"] == "gpt-4o"
    assert "reasoning_effort" not in up.calls[0]["body"]
    assert up.calls[0]["auth"] == "Bearer sk-test-1"
    assert r.headers["x-autopilot-route"] == "passthrough"


def test_auto_routes_easy_to_cheap(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    r = ask(client, text=EASY)
    assert up.calls[0]["body"]["model"] == "gpt-6-luna"
    assert up.calls[0]["body"]["reasoning_effort"] == "none"
    assert r.headers["x-autopilot-route"] == "cheap"
    assert float(r.headers["x-autopilot-p"]) >= 0.5
    assert r.json()["model"] == "gpt-6-luna"


def test_auto_routes_hard_to_expensive(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    r = ask(client, text=HARD)
    assert up.calls[0]["body"]["model"] == "gpt-6-sol"
    assert up.calls[0]["body"]["reasoning_effort"] == "medium"
    assert r.headers["x-autopilot-route"] == "expensive"


def test_content_parts_are_routed(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    body = {"model": "auto",
            "messages": [{"role": "user", "content": [{"type": "text", "text": HARD}]}]}
    client.post(URL, json=body, headers={"Authorization": "Bearer sk-test-1"})
    assert up.calls[0]["body"]["model"] == "gpt-6-sol"


def test_repeat_is_served_from_cache(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    first, second = ask(client), ask(client)
    assert len(up.calls) == 1
    assert second.headers["x-autopilot-route"] == "cache"
    assert second.json() == first.json()


def test_cache_is_isolated_per_key(pipe: Any) -> None:
    up = Upstream()
    client, _ = make(pipe, up)
    ask(client, key="sk-a")
    ask(client, key="sk-b")
    assert len(up.calls) == 2


def test_upstream_error_passed_through(pipe: Any) -> None:
    client, _ = make(pipe, Upstream(status=429))
    r = ask(client)
    assert r.status_code == 429
    assert r.json()["error"]["message"] == "rate limited"


def test_upstream_down_returns_502(pipe: Any) -> None:
    client, _ = make(pipe, Upstream(down=True))
    r = ask(client)
    assert r.status_code == 502
    assert "error" in r.json()


def test_no_key_or_prompt_on_disk(pipe: Any) -> None:
    client, store = make(pipe, Upstream())
    ask(client, text=HARD, key="sk-secret-key")
    dump = "\n".join(store.db.iterdump())
    assert "sk-secret-key" not in dump
    assert HARD not in dump


def test_savings_are_logged(pipe: Any) -> None:
    client, store = make(pipe, Upstream())
    ask(client, text=EASY)
    [row] = store.stats()
    assert row["route"] == "cheap"
    assert row["cost"] == pytest.approx(cost("gpt-6-luna", 100, 50))
    assert row["baseline_cost"] == pytest.approx(cost("gpt-6-sol", 100, 50))

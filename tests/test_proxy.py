import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from autopilot.prices import cost
from autopilot.proxy import create_app
from autopilot.store import Store

URL = "/v1/chat/completions"
AGREE = ["Answer: A"] * 5
SPLIT = ["Answer: A", "Answer: A", "Answer: A", "Answer: B", "Answer: C"]  # agreement 0.6


def completion(model: str, texts: list[str], pt: int, ct: int) -> dict[str, Any]:
    return {
        "id": "x", "object": "chat.completion", "model": model,
        "choices": [{"index": i, "message": {"role": "assistant", "content": t},
                     "finish_reason": "stop"} for i, t in enumerate(texts)],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


class Upstream:
    """Fake OpenAI: the cheap model answers with `cheap`, the expensive one with 'Answer: B'."""

    def __init__(self, cheap: list[str] = AGREE, status: int = 200, down: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.cheap, self.status, self.down = cheap, status, down

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down", request=request)
        body = json.loads(request.content)
        self.calls.append({"body": body, "auth": request.headers["authorization"]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "rate limited"}})
        if body["model"] == "gpt-6-luna":
            n = body.get("n", 1)
            return httpx.Response(200, json=completion("gpt-6-luna", self.cheap[:n], 100, 50 * n))
        return httpx.Response(200, json=completion(body["model"], ["Answer: B"], 100, 200))


def make(upstream: Upstream, threshold: float = 1.0) -> tuple[TestClient, Store]:
    store = Store(":memory:")
    app = create_app(store, threshold, transport=httpx.MockTransport(upstream))
    return TestClient(app), store


def ask(client: TestClient, model: str = "auto", text: str = "Q?", key: str = "sk-test-1",
        **extra: Any) -> Any:  # TestClient returns its own httpx-compatible Response type
    body = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})


def test_missing_key_rejected_before_upstream() -> None:
    up = Upstream()
    client, _ = make(up)
    assert client.post(URL, json={"model": "auto", "messages": []}).status_code == 401
    assert up.calls == []


def test_stream_rejected() -> None:
    up = Upstream()
    client, _ = make(up)
    assert ask(client, stream=True).status_code == 400
    assert up.calls == []


def test_invalid_body_rejected() -> None:
    up = Upstream()
    client, _ = make(up)
    auth = {"Authorization": "Bearer sk-test-1"}
    assert client.post(URL, content=b"not json", headers=auth).status_code == 400
    assert client.post(URL, json=[1, 2], headers=auth).status_code == 400
    assert up.calls == []


def test_n_above_one_rejected_with_auto() -> None:
    up = Upstream()
    client, _ = make(up)
    assert ask(client, n=3).status_code == 400
    assert up.calls == []
    assert ask(client, n=1).status_code == 200


def test_explicit_model_passes_through_untouched() -> None:
    up = Upstream()
    client, _ = make(up)
    r = ask(client, model="gpt-4o")
    assert r.status_code == 200
    assert up.calls[0]["body"] == {"model": "gpt-4o", "messages": [{"role": "user", "content": "Q?"}]}
    assert up.calls[0]["auth"] == "Bearer sk-test-1"
    assert r.headers["x-autopilot-route"] == "passthrough"


def test_agreeing_samples_are_served_from_the_cheap_model() -> None:
    up = Upstream(cheap=AGREE)
    client, _ = make(up)
    r = ask(client)
    assert len(up.calls) == 1
    sent = up.calls[0]["body"]
    assert (sent["model"], sent["n"], sent["reasoning_effort"]) == ("gpt-6-luna", 5, "none")
    assert r.headers["x-autopilot-route"] == "cheap"
    assert r.headers["x-autopilot-agreement"] == "1.00"
    assert len(r.json()["choices"]) == 1
    assert r.json()["choices"][0]["message"]["content"] == "Answer: A"


def test_disagreeing_samples_escalate_to_the_expensive_model() -> None:
    up = Upstream(cheap=SPLIT)
    client, _ = make(up)
    r = ask(client)
    assert len(up.calls) == 2
    sent = up.calls[1]["body"]
    assert (sent["model"], sent["reasoning_effort"]) == ("gpt-6-sol", "medium")
    assert "n" not in sent
    assert r.headers["x-autopilot-route"] == "expensive"
    assert r.headers["x-autopilot-agreement"] == "0.60"
    assert r.json()["choices"][0]["message"]["content"] == "Answer: B"


def test_threshold_is_respected() -> None:
    up = Upstream(cheap=SPLIT)
    client, _ = make(up, threshold=0.6)
    assert ask(client).headers["x-autopilot-route"] == "cheap"
    assert len(up.calls) == 1


def test_repeat_is_served_from_cache() -> None:
    up = Upstream(cheap=SPLIT)
    client, _ = make(up)
    first, second = ask(client), ask(client)
    assert len(up.calls) == 2  # only the first request reached upstream
    assert second.headers["x-autopilot-route"] == "cache"
    assert second.json() == first.json()


def test_cache_is_isolated_per_key() -> None:
    up = Upstream()
    client, _ = make(up)
    ask(client, key="sk-a")
    ask(client, key="sk-b")
    assert len(up.calls) == 2


def test_upstream_error_passed_through() -> None:
    client, store = make(Upstream(status=429))
    r = ask(client)
    assert r.status_code == 429
    assert r.json()["error"]["message"] == "rate limited"
    assert store.stats() == []


def test_upstream_down_returns_502() -> None:
    client, _ = make(Upstream(down=True))
    r = ask(client)
    assert r.status_code == 502
    assert "error" in r.json()


def test_no_key_or_prompt_on_disk() -> None:
    client, store = make(Upstream())
    ask(client, text="a very private prompt", key="sk-secret-key")
    dump = "\n".join(store.db.iterdump())
    assert "sk-secret-key" not in dump
    assert "a very private prompt" not in dump


def test_cheap_route_cost_and_baseline() -> None:
    client, store = make(Upstream(cheap=AGREE))
    ask(client)
    [row] = store.stats()
    assert row["route"] == "cheap"
    assert row["cost"] == pytest.approx(cost("gpt-6-luna", 100, 250))
    assert row["baseline_cost"] == pytest.approx(cost("gpt-6-sol", 100, 50))


def test_escalated_cost_includes_both_calls() -> None:
    client, store = make(Upstream(cheap=SPLIT))
    ask(client)
    [row] = store.stats()
    assert row["route"] == "expensive"
    luna, sol = cost("gpt-6-luna", 100, 250), cost("gpt-6-sol", 100, 200)
    assert luna is not None and sol is not None
    assert row["cost"] == pytest.approx(luna + sol)
    assert row["baseline_cost"] == pytest.approx(cost("gpt-6-sol", 100, 200))

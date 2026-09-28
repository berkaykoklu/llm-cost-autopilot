import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from autopilot.prices import JEV_PRICE, cost
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

    def __init__(self, cheap: list[str] = AGREE, status: int = 200, down: bool = False,
                 jev: float | None = 0.9) -> None:
        self.calls: list[dict[str, Any]] = []
        self.jev_calls: list[dict[str, Any]] = []
        self.cheap, self.status, self.down, self.jev = cheap, status, down, jev

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down", request=request)
        body = json.loads(request.content)
        if request.url.host == "api.typesafe.ai":
            self.jev_calls.append({"body": body, "auth": request.headers["authorization"]})
            if self.jev is None:
                return httpx.Response(500, json={"error": "down"})
            return httpx.Response(200, json={"answers": {"correct": {"type": "noul", "noul": self.jev}},
                                             "usage": {"input_tokens": 300, "output_tokens": 20}})
        self.calls.append({"body": body, "auth": request.headers["authorization"]})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "rate limited"}})
        if body["model"] == "gpt-6-luna":
            n = body.get("n", 1)
            return httpx.Response(200, json=completion("gpt-6-luna", self.cheap[:n], 100, 50 * n))
        return httpx.Response(200, json=completion(body["model"], ["Answer: B"], 100, 200))


def make(upstream: Upstream, threshold: float = 1.0, verifier: str = "consistency"
         ) -> tuple[TestClient, Store]:
    store = Store(":memory:")
    app = create_app(store, threshold, transport=httpx.MockTransport(upstream),
                     verifier=verifier, jev_key="jev-secret-key")
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
    assert r.headers["x-autopilot-score"] == "1.00"
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
    assert r.headers["x-autopilot-score"] == "0.60"
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


def test_sampling_params_are_dropped_on_the_cascade() -> None:
    # temperature 0 would make all 5 samples agree; reasoning Sol rejects temperature/logprobs
    up = Upstream(cheap=SPLIT)
    client, _ = make(up)
    ask(client, temperature=0, top_p=0.1, logprobs=True, top_logprobs=3)
    cheap, exp = up.calls[0]["body"], up.calls[1]["body"]
    assert not {"temperature", "top_p"} & cheap.keys()
    assert not {"temperature", "top_p", "logprobs", "top_logprobs"} & exp.keys()


def test_cache_hit_reuses_the_logged_baseline() -> None:
    client, store = make(Upstream(cheap=AGREE))
    ask(client)
    ask(client)
    stats = {r["route"]: r for r in store.stats()}
    assert stats["cache"]["baseline_cost"] == pytest.approx(stats["cheap"]["baseline_cost"])


def test_null_n_is_accepted() -> None:
    client, _ = make(Upstream())
    assert ask(client, n=None).status_code == 200


def test_zero_choices_with_zero_threshold_does_not_crash() -> None:
    client, _ = make(Upstream(cheap=[]), threshold=0.0)
    assert ask(client).headers["x-autopilot-route"] == "expensive"


def test_non_json_upstream_200_returns_502() -> None:
    client, _ = make(lambda request: httpx.Response(200, content=b"<html>portal</html>"))  # type: ignore[arg-type]
    assert ask(client).status_code == 502


def test_consistency_mode_never_calls_jev() -> None:
    up = Upstream(cheap=SPLIT)
    client, _ = make(up)
    assert ask(client).headers["x-autopilot-verifier"] == "consistency"
    assert up.jev_calls == []


def test_jev_approves_the_cheap_answer() -> None:
    up = Upstream(cheap=["Answer: A"], jev=0.9)
    client, store = make(up, threshold=0.8, verifier="jev")
    r = ask(client, text="Q?")
    assert len(up.calls) == 1
    assert up.calls[0]["body"]["n"] == 1
    [jev] = up.jev_calls
    assert jev["auth"] == "Bearer jev-secret-key"
    assert jev["body"]["state"]["answer"] == "Answer: A"
    assert jev["body"]["state"]["conversation"] == [{"role": "user", "content": "Q?"}]
    assert r.headers["x-autopilot-route"] == "cheap"
    assert r.headers["x-autopilot-score"] == "0.90"
    assert r.headers["x-autopilot-verifier"] == "jev"
    [row] = store.stats()
    luna = cost("gpt-6-luna", 100, 50)
    assert luna is not None
    assert row["cost"] == pytest.approx(luna + 300 * JEV_PRICE / 1_000_000)


def test_jev_rejection_escalates_to_sol() -> None:
    up = Upstream(cheap=["Answer: A"], jev=0.3)
    client, _ = make(up, threshold=0.8, verifier="jev")
    r = ask(client)
    assert r.headers["x-autopilot-route"] == "expensive"
    assert up.calls[1]["body"]["model"] == "gpt-6-sol"


def test_jev_outage_escalates_instead_of_failing() -> None:
    up = Upstream(cheap=["Answer: A"], jev=None)
    client, _ = make(up, threshold=0.8, verifier="jev")
    r = ask(client)
    assert r.status_code == 200
    assert r.headers["x-autopilot-route"] == "expensive"


def test_jev_key_never_on_disk() -> None:
    client, store = make(Upstream(cheap=["Answer: A"]), threshold=0.8, verifier="jev")
    ask(client)
    assert "jev-secret-key" not in "\n".join(store.db.iterdump())

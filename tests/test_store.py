from typing import Any

import pytest

from autopilot.store import Store


def row(route: str, cost: float | None, baseline: float | None) -> dict[str, Any]:
    return {"key_hash": "k", "body_hash": "b", "route": route, "p": None, "model": "m",
            "prompt_tokens": 1, "completion_tokens": 1, "cost": cost, "baseline_cost": baseline}


def test_cache_roundtrip() -> None:
    store = Store(":memory:")
    assert store.get("k", "b") is None
    store.put("k", "b", {"model": "x"})
    assert store.get("k", "b") == {"model": "x"}
    assert store.get("other", "b") is None


def test_stats_group_by_route_and_skip_unpriced() -> None:
    store = Store(":memory:")
    store.log(**row("cheap", 1.0, 5.0))
    store.log(**row("cheap", 2.0, 5.0))
    store.log(**row("passthrough", None, None))
    stats = {r["route"]: r for r in store.stats()}
    assert stats["cheap"]["requests"] == 2
    assert stats["cheap"]["cost"] == pytest.approx(3.0)
    assert stats["cheap"]["baseline_cost"] == pytest.approx(10.0)
    assert stats["passthrough"]["cost"] == 0.0

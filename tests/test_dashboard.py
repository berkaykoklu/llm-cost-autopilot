import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from autopilot.proxy import create_app
from autopilot.store import Store


def client(curve_path: Path) -> TestClient:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    return TestClient(create_app(Store(":memory:"), 1.0, transport, curve_path))


def test_dashboard_page(tmp_path: Path) -> None:
    r = client(tmp_path / "curve.json").get("/")
    assert r.status_code == 200
    assert "LLM Cost Autopilot" in r.text


def test_stats_empty(tmp_path: Path) -> None:
    assert client(tmp_path / "curve.json").get("/api/stats").json() == []


def test_curve_missing_is_404(tmp_path: Path) -> None:
    assert client(tmp_path / "curve.json").get("/api/curve").status_code == 404


def test_curve_served(tmp_path: Path) -> None:
    path = tmp_path / "curve.json"
    path.write_text(json.dumps({"n": 1}))
    assert client(path).get("/api/curve").json() == {"n": 1}

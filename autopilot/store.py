import json
import sqlite3
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key_hash TEXT, body_hash TEXT, response TEXT,
    PRIMARY KEY (key_hash, body_hash)
);
CREATE TABLE IF NOT EXISTS log (
    ts REAL, key_hash TEXT, body_hash TEXT, route TEXT, p REAL, model TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, cost REAL, baseline_cost REAL
);
"""


class Store:
    def __init__(self, path: str) -> None:
        # ponytail: one shared connection, fine for a single-process demo; a pool if it takes real load
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def get(self, key_hash: str, body_hash: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT response FROM cache WHERE key_hash = ? AND body_hash = ?",
                              (key_hash, body_hash)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key_hash: str, body_hash: str, response: dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (key_hash, body_hash, json.dumps(response)))
        self.db.commit()

    def log(self, *, key_hash: str, body_hash: str, route: str, p: float | None, model: str,
            prompt_tokens: int, completion_tokens: int, cost: float | None,
            baseline_cost: float | None) -> None:
        self.db.execute("INSERT INTO log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (time.time(), key_hash, body_hash, route, p, model,
                         prompt_tokens, completion_tokens, cost, baseline_cost))
        self.db.commit()

    def baseline(self, key_hash: str, body_hash: str) -> float | None:
        """What the first, uncached answer to this request would have cost on the expensive model."""
        row = self.db.execute(
            "SELECT baseline_cost FROM log WHERE key_hash = ? AND body_hash = ? AND route != 'cache' "
            "ORDER BY ts DESC LIMIT 1", (key_hash, body_hash)).fetchone()
        return row[0] if row else None

    def stats(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT route, COUNT(*), COALESCE(SUM(cost), 0), COALESCE(SUM(baseline_cost), 0) "
            "FROM log GROUP BY route").fetchall()
        return [{"route": r, "requests": n, "cost": c, "baseline_cost": b} for r, n, c, b in rows]

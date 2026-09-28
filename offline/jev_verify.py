"""Score each held-out cheap answer with Jev, using the exact request the proxy sends (resumable)."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from autopilot import jev
from autopilot.prices import CHEAP
from offline.jsonl import read_jsonl
from offline.label import DATA, EMPTY_ANSWER, request_body
from offline.train import answers, questions

OUT = DATA / "jev_verify.jsonl"


def score(key: str, q: dict[str, Any], answer: str) -> dict[str, Any]:
    body = jev.request_body(request_body(CHEAP, q["prompt"])["messages"], answer)
    r = httpx.post(jev.URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    p, tokens = jev.parse(r.json())
    return {"id": q["id"], "p": p, "tokens": tokens}


def main() -> None:
    key = os.environ["JEV_API_KEY"]
    got = answers(CHEAP)
    done = {r["id"] for r in read_jsonl(OUT)} if OUT.exists() else set()
    todo = [q for q in questions("test") if q["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    with ThreadPoolExecutor(8) as pool, OUT.open("a") as sink:
        for row in pool.map(lambda q: score(key, q, got.get(q["id"], EMPTY_ANSWER)["text"]), todo):
            sink.write(json.dumps(row) + "\n")
            sink.flush()


if __name__ == "__main__":
    main()

"""Sample the cheap model K times per question (resumable), for the self-consistency cascade."""

import asyncio
import json
import math
import os
from typing import Any

import httpx

from autopilot.prices import CHEAP
from offline.grade import extract_letter
from offline.jsonl import read_jsonl
from offline.label import DATA, URL, request_body

K = 5
OUT = DATA / f"samples_{CHEAP}.jsonl"


def letter_confidence(content: list[dict[str, Any]], letter: str | None) -> float:
    """Probability the model gave its answer letter: the last token that is exactly that letter."""
    if not letter:
        return 0.0
    for token in reversed(content):
        if str(token["token"]).strip(" *():.") == letter:
            return math.exp(float(token["logprob"]))
    return 0.0


async def sample(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str,
                 q: dict[str, Any], k: int) -> dict[str, Any]:
    body = {**request_body(CHEAP, q["prompt"]), "logprobs": True}
    r = None
    async with sem:
        for attempt in range(5):
            try:
                r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"},
                                      timeout=120)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2 ** attempt)
    if r is None or r.status_code != 200:
        return {"id": q["id"], "k": k, "letter": None, "conf": 0.0, "pt": 0, "ct": 0,
                "err": r.status_code if r is not None else "network"}
    data = r.json()
    choice = data["choices"][0]
    letter = extract_letter(choice["message"]["content"] or "")
    content = (choice.get("logprobs") or {}).get("content") or []
    return {"id": q["id"], "k": k, "letter": letter, "conf": letter_confidence(content, letter),
            "pt": data["usage"]["prompt_tokens"], "ct": data["usage"]["completion_tokens"]}


async def collect() -> None:
    key = os.environ["OPENAI_API_KEY"]
    done = {(r["id"], r["k"]) for r in read_jsonl(OUT)} if OUT.exists() else set()
    todo = [(q, k) for q in read_jsonl(DATA / "questions.jsonl") for k in range(K)
            if (q["id"], k) not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient() as client:
        with OUT.open("a") as sink:
            for coro in asyncio.as_completed([sample(client, sem, key, q, k) for q, k in todo]):
                sink.write(json.dumps(await coro) + "\n")
                sink.flush()  # resumable: a crash keeps what is already written


def load() -> dict[str, list[dict[str, Any]]]:
    """Samples per question id, ordered by k; sample 0 is the one the cascade would serve."""
    by_id: dict[str, dict[int, dict[str, Any]]] = {}
    for row in read_jsonl(OUT):
        by_id.setdefault(row["id"], {})[row["k"]] = row
    return {qid: [ks[k] for k in sorted(ks)] for qid, ks in by_id.items()}


if __name__ == "__main__":
    asyncio.run(collect())

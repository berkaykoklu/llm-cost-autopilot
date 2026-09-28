import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from autopilot.consistency import agreement
from autopilot.prices import CHEAP, EXPENSIVE, REASONING_EFFORT, cost
from autopilot.store import Store

UPSTREAM = "https://api.openai.com/v1/chat/completions"
# the cascade needs varied samples, and reasoning models reject these when effort is not "none"
CHEAP_DROP = ("temperature", "top_p")
EXPENSIVE_DROP = ("n", "temperature", "top_p", "logprobs", "top_logprobs")
HERE = Path(__file__).parent
SAMPLES = 5


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "autopilot_error"}},
                        status_code=status)


def usage_of(data: dict[str, Any]) -> tuple[int, int]:
    usage = data.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def create_app(store: Store, threshold: float,
               transport: httpx.AsyncBaseTransport | None = None,
               curve_path: Path = Path("data/curve.json")) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(transport=transport, timeout=120)

    async def call(body: dict[str, Any], auth: str) -> tuple[dict[str, Any], Response | None]:
        """The upstream JSON, or the response to hand back unchanged when upstream failed."""
        try:
            r = await client.post(UPSTREAM, json=body, headers={"Authorization": auth})
        except httpx.HTTPError as exc:
            return {}, error(502, f"Upstream request failed: {type(exc).__name__}")
        if r.status_code != 200:
            return {}, Response(r.content, status_code=r.status_code,
                                media_type=r.headers.get("content-type"))
        try:
            return r.json(), None
        except ValueError:
            return {}, error(502, "Upstream returned a non-JSON response")

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or not auth[7:].strip():
            return error(401, "Missing API key: send your OpenAI key as 'Authorization: Bearer ...'")
        try:
            body = await request.json()
        except ValueError:
            return error(400, "Request body must be JSON")
        if not isinstance(body, dict):
            return error(400, "Request body must be a JSON object")
        if body.get("stream"):
            return error(400, "Streaming is not supported yet; send stream=false")
        routed = body.get("model") == "auto"
        if routed and body.get("n") not in (None, 1):
            return error(400, "model=auto returns one answer; send n=1 or omit it")

        key_hash = sha(auth)  # the key itself never touches disk
        body_hash = sha(json.dumps(body, sort_keys=True))

        cached = store.get(key_hash, body_hash)
        if cached is not None:
            full = store.baseline(key_hash, body_hash)
            store.log(key_hash=key_hash, body_hash=body_hash, route="cache", p=None,
                      model=str(cached.get("model", "")), prompt_tokens=0, completion_tokens=0,
                      cost=0.0, baseline_cost=full)
            return JSONResponse(cached, headers={"x-autopilot-route": "cache"})

        if not routed:
            data, failed = await call(body, auth)
            if failed:
                return failed
            pt, ct = usage_of(data)
            actual = cost(str(body.get("model", "")), pt, ct)
            store.put(key_hash, body_hash, data)
            store.log(key_hash=key_hash, body_hash=body_hash, route="passthrough", p=None,
                      model=str(data.get("model", "")), prompt_tokens=pt, completion_tokens=ct,
                      cost=actual, baseline_cost=actual)
            return JSONResponse(data, headers={"x-autopilot-route": "passthrough"})

        # cascade: ask the cheap model several times; if it keeps giving the same answer, trust it
        cheap_body = {k: v for k, v in body.items() if k not in CHEAP_DROP}
        cheap_body |= {"model": CHEAP, "n": SAMPLES, "reasoning_effort": REASONING_EFFORT[CHEAP]}
        cheap, failed = await call(cheap_body, auth)
        if failed:
            return failed
        texts = [(c.get("message") or {}).get("content") or "" for c in cheap.get("choices") or []]
        agree = agreement(texts)
        cpt, cct = usage_of(cheap)
        cheap_cost = cost(CHEAP, cpt, cct) or 0.0

        if texts and agree >= threshold:
            data = {**cheap, "choices": [{**cheap["choices"][0], "index": 0}]}
            route, pt, ct, actual = "cheap", cpt, cct, cheap_cost
            # ponytail: assumes Sol would have used one sample's tokens; it reasons, so this undercounts
            baseline = cost(EXPENSIVE, cpt, cct // max(len(texts), 1))
        else:
            exp_body = {k: v for k, v in body.items() if k not in EXPENSIVE_DROP}
            exp_body |= {"model": EXPENSIVE, "reasoning_effort": REASONING_EFFORT[EXPENSIVE]}
            data, failed = await call(exp_body, auth)
            if failed:
                return failed
            pt, ct = usage_of(data)
            baseline = cost(EXPENSIVE, pt, ct)
            route, actual = "expensive", cheap_cost + (baseline or 0.0)

        store.put(key_hash, body_hash, data)
        store.log(key_hash=key_hash, body_hash=body_hash, route=route, p=agree,
                  model=str(data.get("model", "")), prompt_tokens=pt, completion_tokens=ct,
                  cost=actual, baseline_cost=baseline)
        return JSONResponse(data, headers={"x-autopilot-route": route,
                                           "x-autopilot-agreement": f"{agree:.2f}"})

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(HERE / "dashboard.html")

    @app.get("/api/stats")
    def stats() -> list[dict[str, Any]]:
        return store.stats()

    @app.get("/api/curve")
    def curve() -> Response:
        if not curve_path.exists():
            return error(404, "No offline result yet: run python -m offline.curve")
        return Response(curve_path.read_text(), media_type="application/json")

    return app


def main_app() -> FastAPI:  # uvicorn autopilot.proxy:main_app --factory
    return create_app(
        Store(os.environ.get("AUTOPILOT_DB", "autopilot.db")),
        float(os.environ.get("AUTOPILOT_THRESHOLD", "1.0")),  # 1.0 = all samples must agree
        curve_path=Path(os.environ.get("AUTOPILOT_CURVE", "data/curve.json")),
    )

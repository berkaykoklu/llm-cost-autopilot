import hashlib
import json
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from autopilot import router
from autopilot.prices import CHEAP, EXPENSIVE, REASONING_EFFORT, cost
from autopilot.store import Store

UPSTREAM = "https://api.openai.com/v1/chat/completions"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "autopilot_error"}},
                        status_code=status)


def create_app(store: Store, pipe: Any, threshold: float,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(transport=transport, timeout=120)

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

        key_hash = sha(auth)  # the key itself never touches disk
        body_hash = sha(json.dumps(body, sort_keys=True))
        routed = body.get("model") == "auto"

        cached = store.get(key_hash, body_hash)
        if cached is not None:
            usage = cached.get("usage") or {}
            full = cost(EXPENSIVE if routed else str(cached.get("model", "")),
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            store.log(key_hash=key_hash, body_hash=body_hash, route="cache", p=None,
                      model=str(cached.get("model", "")), prompt_tokens=0, completion_tokens=0,
                      cost=0.0, baseline_cost=full)
            return JSONResponse(cached, headers={"x-autopilot-route": "cache"})

        headers = {"x-autopilot-route": "passthrough"}
        p = None
        if routed:
            p = router.p_cheap(pipe, router.prompt_text(body.get("messages")))
            model = router.choose(p, threshold)
            headers = {"x-autopilot-route": "cheap" if model == CHEAP else "expensive",
                       "x-autopilot-p": f"{p:.3f}"}
            # serve with the same settings the cost–accuracy curve was measured with
            body = {**body, "model": model, "reasoning_effort": REASONING_EFFORT[model]}

        try:
            upstream = await client.post(UPSTREAM, json=body, headers={"Authorization": auth})
        except httpx.HTTPError as exc:
            return error(502, f"Upstream request failed: {type(exc).__name__}")
        if upstream.status_code != 200:
            return Response(upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"))

        data = upstream.json()
        usage = data.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        actual = cost(str(body.get("model", "")), pt, ct)
        # ponytail: assumes Sol would have used the same tokens; a true counterfactual needs a paired run
        baseline = cost(EXPENSIVE, pt, ct) if routed else actual
        store.put(key_hash, body_hash, data)
        store.log(key_hash=key_hash, body_hash=body_hash, route=headers["x-autopilot-route"],
                  p=p, model=str(data.get("model", "")), prompt_tokens=pt, completion_tokens=ct,
                  cost=actual, baseline_cost=baseline)
        return JSONResponse(data, headers=headers)

    return app

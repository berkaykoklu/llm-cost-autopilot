"""TypeSafe Jev as a verifier: is the cheap model's answer correct? Shared by the proxy and the offline eval."""

from typing import Any

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": "`answer` is a correct reply to the last user message in `conversation`.",
    },
}


def request_body(messages: object, answer: str) -> dict[str, Any]:
    conversation = [{"role": m.get("role"), "content": m.get("content")}
                    for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []
    return {"model": MODEL, "questions": QUESTIONS,
            "state": {"conversation": conversation, "answer": answer}}


def parse(data: dict[str, Any]) -> tuple[float | None, int]:
    """(probability the answer is correct, input tokens billed)."""
    answer = (data.get("answers") or {}).get("correct") or {}
    score = answer.get("noul")
    tokens = (data.get("usage") or {}).get("input_tokens", 0)
    return (float(score) if isinstance(score, int | float) else None), int(tokens)

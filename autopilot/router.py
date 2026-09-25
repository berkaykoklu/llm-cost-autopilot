from pathlib import Path
from typing import Any

import joblib

from autopilot.prices import CHEAP, EXPENSIVE


def load(path: str | Path) -> Any:  # a fitted sklearn Pipeline; sklearn ships no type stubs
    return joblib.load(path)


def p_cheap(pipe: Any, text: str) -> float:
    # column 1 = label 1 = "the cheap model was right"
    return float(pipe.predict_proba([text])[0][1])


def choose(p: float, threshold: float) -> str:
    return CHEAP if p >= threshold else EXPENSIVE


def prompt_text(messages: object) -> str:
    """User-message text only, which matches what the router was trained on."""
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [c["text"] for c in content
                      if isinstance(c, dict) and isinstance(c.get("text"), str)]
    return "\n".join(parts)

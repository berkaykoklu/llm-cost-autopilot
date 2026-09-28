"""The prompt-only TF-IDF router. Kept as the offline baseline the cascade is measured against."""

from pathlib import Path
from typing import Any

import joblib


def load(path: str | Path) -> Any:  # a fitted sklearn Pipeline; sklearn ships no type stubs
    return joblib.load(path)


def p_cheap(pipe: Any, text: str) -> float:
    # column 1 = label 1 = "the cheap model was right"
    return float(pipe.predict_proba([text])[0][1])

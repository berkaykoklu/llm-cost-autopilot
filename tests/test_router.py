from pathlib import Path
from typing import Any

import pytest

from autopilot.router import choose, load, p_cheap, prompt_text
from offline.train import fit, top_words

EASY = ["what colour is the sky", "what colour is grass"]
HARD = ["prove the theorem about eigenvalues", "prove the lemma about eigenvalues"]


def toy_pipe() -> Any:
    return fit((EASY + HARD) * 5, [1, 1, 0, 0] * 5)


def test_easy_scores_higher_than_hard() -> None:
    pipe = toy_pipe()
    assert p_cheap(pipe, EASY[0]) > 0.5 > p_cheap(pipe, HARD[0])


def test_choose_threshold() -> None:
    assert choose(0.7, 0.5) == "gpt-6-luna"
    assert choose(0.5, 0.5) == "gpt-6-luna"
    assert choose(0.3, 0.5) == "gpt-6-sol"


def test_top_words() -> None:
    words = top_words(toy_pipe(), 3)  # prove/about/eigenvalues tie on weight
    assert "prove" in words["hard"]
    assert "colour" in words["easy"]


def test_prompt_text_string_and_parts() -> None:
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image_url"}]},
    ]
    assert prompt_text(messages) == "first\nsecond"


def test_prompt_text_tolerates_garbage() -> None:
    assert prompt_text(None) == ""
    assert prompt_text([1, "x", {"role": "user"}]) == ""


def test_load_missing_router_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.pkl")

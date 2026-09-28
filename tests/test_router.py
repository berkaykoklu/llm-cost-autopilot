from pathlib import Path
from typing import Any

import pytest

from autopilot.router import load, p_cheap
from offline.train import fit, top_words

EASY = ["what colour is the sky", "what colour is grass"]
HARD = ["prove the theorem about eigenvalues", "prove the lemma about eigenvalues"]


def toy_pipe() -> Any:
    return fit((EASY + HARD) * 5, [1, 1, 0, 0] * 5)


def test_easy_scores_higher_than_hard() -> None:
    pipe = toy_pipe()
    assert p_cheap(pipe, EASY[0]) > 0.5 > p_cheap(pipe, HARD[0])



def test_top_words() -> None:
    words = top_words(toy_pipe(), 3)  # prove/about/eigenvalues tie on weight
    assert "prove" in words["hard"]
    assert "colour" in words["easy"]




def test_load_missing_router_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.pkl")

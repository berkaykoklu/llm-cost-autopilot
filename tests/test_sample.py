from collections import Counter
from pathlib import Path
from typing import Any

from offline.jsonl import read_jsonl, write_jsonl
from offline.sample import format_prompt, split, stratified


def rows() -> list[dict[str, Any]]:
    return [{"id": f"{c}{i}", "category": c} for c in "xyz" for i in range(10)]


def test_format_prompt() -> None:
    assert format_prompt("Q?", ["one", "two"]) == "Q?\n\nA. one\nB. two"


def test_stratified_takes_evenly_from_each_category() -> None:
    picked = stratified(rows(), 9, seed=1)
    assert Counter(r["category"] for r in picked) == {"x": 3, "y": 3, "z": 3}


def test_stratified_is_deterministic() -> None:
    assert stratified(rows(), 9, seed=1) == stratified(rows(), 9, seed=1)


def test_split_sizes() -> None:
    out = split(rows(), 20, seed=1)
    assert Counter(r["split"] for r in out) == {"train": 20, "test": 10}


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "a.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": "x"}])
    assert read_jsonl(path) == [{"a": 1}, {"b": "x"}]

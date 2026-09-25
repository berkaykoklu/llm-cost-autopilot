import random
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from offline.jsonl import write_jsonl

LETTERS = "ABCDEFGHIJ"
N_TOTAL, N_TRAIN, SEED = 1000, 700, 42
OUT = Path("data/questions.jsonl")


def format_prompt(question: str, options: list[str]) -> str:
    return question + "\n\n" + "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def stratified(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    ordered = [groups[key] for key in sorted(groups)]
    for group in ordered:
        rng.shuffle(group)
    # round-robin across categories, so every subject gets an equal share
    return [row for tier in zip_longest(*ordered) for row in tier if row is not None][:n]


def split(rows: list[dict[str, Any]], n_train: int, seed: int) -> list[dict[str, Any]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return [{**row, "split": "train" if i < n_train else "test"} for i, row in enumerate(shuffled)]


def main() -> None:
    from datasets import load_dataset

    rows = []
    for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        assert r["answer"] in LETTERS[: len(r["options"])], f"bad gold in {r['question_id']}"
        rows.append(
            {
                "id": str(r["question_id"]),
                "category": r["category"],
                "prompt": format_prompt(r["question"], r["options"]),
                "gold": r["answer"],
            }
        )
    out = split(stratified(rows, N_TOTAL, SEED), N_TRAIN, SEED)
    write_jsonl(OUT, out)
    print(f"wrote {len(out)} questions to {OUT}")


if __name__ == "__main__":
    main()

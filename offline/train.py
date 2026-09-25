from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from autopilot.prices import CHEAP
from offline.grade import is_correct
from offline.jsonl import read_jsonl
from offline.label import EMPTY_ANSWER

DATA = Path("data")
ROUTER = DATA / "router.pkl"


def fit(texts: list[str], labels: list[int]) -> Any:
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2),
                         LogisticRegression(max_iter=1000))
    return pipe.fit(texts, labels)


def top_words(pipe: Any, k: int) -> dict[str, list[str]]:
    names = pipe[0].get_feature_names_out()
    order = pipe[-1].coef_[0].argsort()  # negative weight = pushes toward "cheap model fails"
    return {"easy": [str(names[i]) for i in order[::-1][:k]],
            "hard": [str(names[i]) for i in order[:k]]}


def questions(split: str) -> list[dict[str, Any]]:
    return [q for q in read_jsonl(DATA / "questions.jsonl") if q["split"] == split]


def answers(model: str) -> dict[str, dict[str, Any]]:
    return {a["id"]: a for a in read_jsonl(DATA / f"answers_{model}.jsonl")}


def main() -> None:
    qs, got = questions("train"), answers(CHEAP)
    labels = [int(is_correct(got.get(q["id"], EMPTY_ANSWER)["text"], q["gold"])) for q in qs]
    pipe = fit([q["prompt"] for q in qs], labels)
    joblib.dump(pipe, ROUTER)
    missing = sum(q["id"] not in got for q in qs)
    print(f"train: {len(qs)} questions, cheap model right on {sum(labels)}, missing {missing}")
    print(top_words(pipe, 15))


if __name__ == "__main__":
    main()

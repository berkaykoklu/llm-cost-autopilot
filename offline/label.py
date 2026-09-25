import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from autopilot.prices import CHEAP, EXPENSIVE, cost
from offline.grade import extract_letter, is_correct
from offline.jsonl import read_jsonl, write_jsonl

SYSTEM = (
    "Answer the multiple-choice question. Think step by step, then end with a final line "
    "'Answer: X', where X is the option letter."
)
URL = "https://api.openai.com/v1/chat/completions"
DATA = Path("data")
BUDGET_USD = 5.0
BATCH_DISCOUNT = 0.5
PILOT_N = 20
EMPTY_ANSWER: dict[str, Any] = {"text": "", "prompt_tokens": 0, "completion_tokens": 0}


def request_body(model: str, prompt: str) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    # hidden reasoning tokens are billed as output; "none" keeps cost predictable
    return {"model": model, "messages": messages, "reasoning_effort": "none"}


def request_line(qid: str, model: str, prompt: str) -> dict[str, Any]:
    return {"custom_id": qid, "method": "POST", "url": "/v1/chat/completions",
            "body": request_body(model, prompt)}


def parse_output_line(line: dict[str, Any]) -> dict[str, Any]:
    response = line.get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") != 200 or not body.get("choices"):
        return {"id": line["custom_id"], **EMPTY_ANSWER}  # a failed request grades as wrong
    usage = body.get("usage") or {}
    return {
        "id": line["custom_id"],
        "text": body["choices"][0]["message"].get("content") or "",
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def targets(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # labels need only the cheap model; the expensive one is needed only to score the test split
    return {CHEAP: questions, EXPENSIVE: [q for q in questions if q["split"] == "test"]}


def projected_cost(mean_cost: dict[str, float], counts: dict[str, int]) -> float:
    return BATCH_DISCOUNT * sum(mean_cost[m] * counts[m] for m in counts)


def guard_submit(pilot_path: Path, batches_path: Path) -> None:
    if not pilot_path.exists():
        sys.exit("run the pilot first: python -m offline.label pilot")
    projected = json.loads(pilot_path.read_text())["projected_batch_usd"]
    if projected > BUDGET_USD:
        sys.exit(f"projected ${projected:.2f} is over the ${BUDGET_USD} budget; not submitting")
    if batches_path.exists():
        sys.exit(f"already submitted ({batches_path}); delete it to submit again")


def pilot() -> None:
    key = os.environ["OPENAI_API_KEY"]
    questions = read_jsonl(DATA / "questions.jsonl")
    sample = [q for q in questions if q["split"] == "test"][:PILOT_N]
    mean_cost: dict[str, float] = {}
    for model in (CHEAP, EXPENSIVE):
        answers = []
        for q in sample:
            r = httpx.post(URL, json=request_body(model, q["prompt"]),
                           headers={"Authorization": f"Bearer {key}"}, timeout=120)
            if r.status_code != 200:
                sys.exit(f"{model}: HTTP {r.status_code} {r.text}")
            answers.append(parse_output_line(
                {"custom_id": q["id"], "response": {"status_code": 200, "body": r.json()}}))
        costs = [cost(model, a["prompt_tokens"], a["completion_tokens"]) or 0.0 for a in answers]
        mean_cost[model] = sum(costs) / len(costs)
        correct = sum(is_correct(a["text"], q["gold"]) for a, q in zip(answers, sample))
        unparsed = sum(extract_letter(a["text"]) is None for a in answers)
        tokens = sum(a["completion_tokens"] for a in answers) / len(answers)
        print(f"{model}: correct {correct}/{len(sample)}, unparseable {unparsed}, "
              f"mean completion tokens {tokens:.0f}, mean ${mean_cost[model]:.5f}/question")
    counts = {m: len(qs) for m, qs in targets(questions).items()}
    projected = projected_cost(mean_cost, counts)
    (DATA / "pilot.json").write_text(json.dumps(
        {"mean_cost": mean_cost, "projected_batch_usd": projected}, indent=1))
    print(f"projected batch cost ${projected:.2f} (budget ${BUDGET_USD})")


def submit() -> None:
    guard_submit(DATA / "pilot.json", DATA / "batches.json")
    client = OpenAI()
    ids = {}
    for model, qs in targets(read_jsonl(DATA / "questions.jsonl")).items():
        path = DATA / f"batch_input_{model}.jsonl"  # the Batch API takes one model per file
        write_jsonl(path, [request_line(q["id"], model, q["prompt"]) for q in qs])
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id,
                                      endpoint="/v1/chat/completions", completion_window="24h")
        ids[model] = batch.id
        print(f"{model}: batch {batch.id}, {len(qs)} requests")
    (DATA / "batches.json").write_text(json.dumps(ids, indent=1))


def collect() -> None:
    client = OpenAI()
    for model, batch_id in json.loads((DATA / "batches.json").read_text()).items():
        batch = client.batches.retrieve(batch_id)
        print(f"{model}: {batch.status} {batch.request_counts}")
        if batch.status != "completed" or batch.output_file_id is None:
            continue
        text = client.files.content(batch.output_file_id).text
        answers = [parse_output_line(json.loads(line)) for line in text.splitlines() if line]
        write_jsonl(DATA / f"answers_{model}.jsonl", answers)
        spent = BATCH_DISCOUNT * sum(
            cost(model, a["prompt_tokens"], a["completion_tokens"]) or 0.0 for a in answers)
        empty = sum(a["text"] == "" for a in answers)
        print(f"  saved {len(answers)} answers ({empty} empty), cost ${spent:.2f}")


if __name__ == "__main__":
    {"pilot": pilot, "submit": submit, "collect": collect}[sys.argv[1]]()

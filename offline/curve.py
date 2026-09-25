import json
from pathlib import Path

from autopilot import router
from autopilot.prices import CHEAP, EXPENSIVE, cost
from offline.grade import extract_letter, is_correct
from offline.label import EMPTY_ANSWER
from offline.train import ROUTER, answers, questions, top_words

CURVE = Path("data/curve.json")
Point = dict[str, float | None]


def sweep(scores: list[float], cheap_ok: list[bool], exp_ok: list[bool],
          cheap_cost: list[float], exp_cost: list[float]) -> list[Point]:
    """Send the k highest-scoring questions to the cheap model, for k = 0..n."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    points: list[Point] = []
    for k in range(n + 1):
        cheap = set(order[:k])  # ponytail: O(n²), fine for n = 300
        points.append({
            "frac_cheap": k / n,
            "cost": sum(cheap_cost[i] if i in cheap else exp_cost[i] for i in range(n)),
            "accuracy": sum(cheap_ok[i] if i in cheap else exp_ok[i] for i in range(n)) / n,
            "threshold": scores[order[k - 1]] if k else None,
        })
    return points


def recommend(points: list[Point], tolerance: float = 0.02) -> Point:
    best = points[0]["accuracy"] or 0.0  # k = 0: everything to the expensive model
    ok = [p for p in points if (p["accuracy"] or 0.0) >= best - tolerance]
    return min(ok, key=lambda p: p["cost"] or 0.0)


def main() -> None:
    qs = questions("test")
    by_model = {m: answers(m) for m in (CHEAP, EXPENSIVE)}
    got = {m: [a.get(q["id"], EMPTY_ANSWER) for q in qs] for m, a in by_model.items()}
    ok = {m: [is_correct(a["text"], q["gold"]) for a, q in zip(got[m], qs)] for m in got}
    spend = {m: [cost(m, a["prompt_tokens"], a["completion_tokens"]) or 0.0 for a in got[m]]
             for m in got}
    pipe = router.load(ROUTER)
    args = (ok[CHEAP], ok[EXPENSIVE], spend[CHEAP], spend[EXPENSIVE])
    points = sweep([router.p_cheap(pipe, q["prompt"]) for q in qs], *args)
    result = {
        "n": len(qs),
        "router": points,
        "length_rule": sweep([-float(len(q["prompt"])) for q in qs], *args),  # shorter = easier
        "top_words": top_words(pipe, 10),
        "unparseable": {m: sum(extract_letter(a["text"]) is None for a in got[m]) for m in got},
    }
    CURVE.write_text(json.dumps(result, indent=1))
    print(f"all expensive: {points[0]}\nall cheap: {points[-1]}")
    print(f"recommended (within 2 points of all-expensive): {recommend(points)}")


if __name__ == "__main__":
    main()

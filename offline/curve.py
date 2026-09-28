import json
from pathlib import Path

from autopilot import router
from autopilot.consistency import agreement_of_keys
from autopilot.prices import CHEAP, EXPENSIVE, JEV_PRICE, cost
from offline import consistency
from offline.grade import extract_letter, is_correct
from offline.jev_verify import OUT as JEV_SCORES
from offline.jsonl import read_jsonl
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


def cascade_by_threshold(agree: list[float], cheap_ok: list[bool], exp_ok: list[bool],
                         cheap_cost: list[float], exp_cost: list[float],
                         thresholds: list[float]) -> list[dict[str, float]]:
    """Serve the cheap answer when agreement >= t, else escalate (exp_cost includes the cheap call)."""
    n = len(agree)
    rows = []
    for t in thresholds:
        up = [a < t for a in agree]
        rows.append({
            "threshold": t,
            "escalated": sum(up) / n,
            "accuracy": sum(exp_ok[i] if up[i] else cheap_ok[i] for i in range(n)) / n,
            "cost": sum(exp_cost[i] if up[i] else cheap_cost[i] for i in range(n)),
        })
    return rows


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
    samples = consistency.load()
    # the proxy asks for all 5 samples in one call (n=5), so the prompt is billed once
    cheap5 = [cost(CHEAP, samples[q["id"]][0]["pt"], sum(s["ct"] for s in samples[q["id"]])) or 0.0
              for q in qs]
    agree = [agreement_of_keys([s["letter"] or "" for s in samples[q["id"]]]) for q in qs]
    served_ok = [samples[q["id"]][0]["letter"] == q["gold"] for q in qs]
    escalated_cost = [c + e for c, e in zip(cheap5, spend[EXPENSIVE])]
    cascade_args = (served_ok, ok[EXPENSIVE], cheap5, escalated_cost)
    table = cascade_by_threshold(agree, *cascade_args, thresholds=[0.2, 0.4, 0.6, 0.8, 1.0])
    result = {
        "n": len(qs),
        "router": points,
        "cascade": sweep(agree, *cascade_args),
        "cascade_by_threshold": table,
        "length_rule": sweep([-float(len(q["prompt"])) for q in qs], *args),  # shorter = easier
        "top_words": top_words(pipe, 10),
        "unparseable": {m: sum(extract_letter(a["text"]) is None for a in got[m]) for m in got},
    }
    if JEV_SCORES.exists():
        jev = {r["id"]: r for r in read_jsonl(JEV_SCORES)}
        jev_cost = [spend[CHEAP][i] + jev[q["id"]]["tokens"] * JEV_PRICE / 1_000_000
                    for i, q in enumerate(qs)]
        jev_args = (ok[CHEAP], ok[EXPENSIVE], jev_cost, [c + e for c, e in zip(jev_cost, spend[EXPENSIVE])])
        jev_p = [jev[q["id"]]["p"] or 0.0 for q in qs]
        jev_table = cascade_by_threshold(jev_p, *jev_args, thresholds=[0.5, 0.8, 0.9, 0.95])
        result |= {"jev": sweep(jev_p, *jev_args), "jev_by_threshold": jev_table}
        for row in jev_table:
            print(f"jev {row}")
    CURVE.write_text(json.dumps(result, indent=1))
    print(f"all expensive: {points[0]}\nall cheap: {points[-1]}")
    print(f"recommended (within 2 points of all-expensive): {recommend(points)}")
    for row in table:
        print(f"cascade {row}")


if __name__ == "__main__":
    main()

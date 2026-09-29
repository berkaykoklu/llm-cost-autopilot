"""The numbers the portfolio page shows, from the same code that produced the README.

Run: uv run python -m offline.site_export   (writes data/site.json)
"""

import json
from pathlib import Path
from typing import Any

from autopilot import router
from autopilot.consistency import agreement_of_keys
from autopilot.prices import CHEAP, EXPENSIVE, cost
from offline import consistency
from offline.grade import is_correct
from offline.jev_verify import OUT as JEV_SCORES
from offline.jsonl import read_jsonl
from offline.label import EMPTY_ANSWER
from offline.train import ROUTER, answers, questions

DATA = Path(__file__).parent.parent / "data"
OUT = DATA / "site.json"
# The README's figures; the export refuses to write anything that disagrees.
README = {
    "sol_acc": 0.700,
    "sol_cost": 0.769,
    "cascade_acc": 0.673,
    "cascade_cost": 0.491,
    "router_acc_at_cascade_cost": 0.637,
    "luna_acc": 0.557,
    "luna_cost": 0.013,
}
README_AUC = {"tfidf": 0.52, "agree": 0.66, "jev": 0.72}


def auc(scores: list[float], labels: list[bool]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_at(points: list[dict[str, Any]], budget: float) -> float:
    return max((float(p["accuracy"]) for p in points if p["cost"] <= budget + 1e-12), default=0.0)


def build() -> dict[str, Any]:
    qs = questions("test")
    sol, luna = answers(EXPENSIVE), answers(CHEAP)
    samples = consistency.load()
    pipe = router.load(ROUTER)
    jev = {r["id"]: float(r["p"]) for r in read_jsonl(JEV_SCORES)} if JEV_SCORES.exists() else {}
    curve = json.loads((DATA / "curve.json").read_text())

    trace: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    for q in qs:
        s = samples[q["id"]]
        letters = [x["letter"] or "" for x in s]
        agree = agreement_of_keys(letters)
        # the proxy asks for all 5 samples in one call (n=5), so the prompt is billed once
        cheap5 = cost(CHEAP, s[0]["pt"], sum(x["ct"] for x in s)) or 0.0
        a = sol.get(q["id"], EMPTY_ANSWER)
        sol_cost = cost(EXPENSIVE, a["prompt_tokens"], a["completion_tokens"]) or 0.0
        sol_ok = is_correct(a["text"], q["gold"])
        luna_ok = is_correct(luna.get(q["id"], EMPTY_ANSWER)["text"], q["gold"])
        first_ok = letters[0] == q["gold"]
        up = agree < 1.0
        trace.append(
            {
                "subject": q["category"],
                "q": q["prompt"].split("\n\n")[0][:170],
                "gold": q["gold"],
                "letters": letters,
                "route": "sol" if up else "luna",
                "ok": sol_ok if up else first_ok,
                "cost": round(cheap5 + (sol_cost if up else 0.0), 7),
                "sol_cost": round(sol_cost, 7),
                "sol_ok": sol_ok,
                "luna_ok": luna_ok,
            }
        )
        signals.append(
            {
                "tfidf": round(float(router.p_cheap(pipe, q["prompt"])), 5),
                "agree": agree,
                "jev": jev.get(q["id"]),
                "luna_ok": luna_ok,
                "first_ok": first_ok,
            }
        )

    n = len(trace)
    jev_t = float(curve["jev_threshold_from_train"])
    wrong = [s for s in signals if s["agree"] == 1.0 and not s["first_ok"]]
    fixed_jev = next(r for r in curve["jev_by_threshold"] if abs(r["threshold"] - jev_t) < 1e-9)
    spent = sum(t["cost"] for t in trace)
    headline = {
        "n": n,
        "sol_acc": sum(t["sol_ok"] for t in trace) / n,
        "sol_cost": sum(t["sol_cost"] for t in trace),
        "cascade_acc": sum(t["ok"] for t in trace) / n,
        "cascade_cost": spent,
        "router_acc_at_cascade_cost": best_at(curve["router"], spent),
        "luna_acc": sum(t["luna_ok"] for t in trace) / n,
        "luna_cost": curve["router"][-1]["cost"],
        "escalated": sum(t["route"] == "sol" for t in trace),
        "agree_all": sum(s["agree"] == 1.0 for s in signals),
        "agree_all_wrong": len(wrong),
        "agree_all_wrong_jev_pass": sum((s["jev"] or 0.0) >= jev_t for s in wrong),
        "jev_threshold": jev_t,
        "auc": {
            "tfidf": auc([s["tfidf"] for s in signals], [s["luna_ok"] for s in signals]),
            "agree": auc([s["agree"] for s in signals], [s["first_ok"] for s in signals]),
            "jev": auc([s["jev"] or 0.0 for s in signals], [s["luna_ok"] for s in signals]),
        },
    }

    def pts(key: str) -> list[dict[str, float]]:
        return [{"c": round(p["cost"], 5), "a": round(p["accuracy"], 5)} for p in curve[key]]

    return {
        "trace": trace,
        "signals": signals,
        "headline": headline,
        "curves": {
            "cascade": pts("cascade"),
            "jev": pts("jev"),
            "router": pts("router"),
            "length": pts("length_rule"),
        },
        "fixed": {
            "cascade": {"c": spent, "a": headline["cascade_acc"], "rule": "all 5 agree"},
            "jev": {
                "c": fixed_jev["cost"],
                "a": fixed_jev["accuracy"],
                "rule": f"Youden on train, {jev_t}",
            },
        },
    }


def check(site: dict[str, Any]) -> None:
    h = site["headline"]
    bad = [k for k, v in README.items() if round(h[k], 3) != v]
    bad += [f"auc.{k}" for k, v in README_AUC.items() if round(h["auc"][k], 2) != v]
    if bad:
        raise SystemExit(f"site export disagrees with the README on: {', '.join(bad)}")


def main() -> None:
    site = build()
    check(site)
    OUT.write_text(json.dumps(site, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)", json.dumps(site["headline"]))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pytest

from offline.site_export import auc, build, check

CURVE = json.loads((Path(__file__).parent.parent / "data" / "curve.json").read_text())


def test_auc_ranks() -> None:
    assert auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auc([0.1, 0.9], [True, False]) == 0.0
    assert auc([0.5, 0.5], [True, False]) == 0.5


def test_headline_matches_curve_json() -> None:
    h = build()["headline"]
    cascade = next(r for r in CURVE["cascade_by_threshold"] if r["threshold"] == 1.0)
    assert h["cascade_acc"] == pytest.approx(cascade["accuracy"], abs=1e-9)
    assert h["cascade_cost"] == pytest.approx(cascade["cost"], abs=1e-6)
    assert h["sol_acc"] == pytest.approx(CURVE["router"][0]["accuracy"], abs=1e-9)
    assert h["n"] == 300


def test_check_rejects_a_drifted_headline() -> None:
    site = build()
    site["headline"]["cascade_acc"] = 0.5
    with pytest.raises(SystemExit):
        check(site)


def test_check_rejects_a_drifted_jev_pin() -> None:
    site = build()
    site["fixed"]["jev"]["a"] = 0.70  # the hindsight ceiling, not the result fixed in advance
    with pytest.raises(SystemExit):
        check(site)

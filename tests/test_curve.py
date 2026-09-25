import pytest

from offline.curve import recommend, sweep


def toy() -> list[dict[str, float | None]]:
    # item 0: cheap is right and scores high; item 1: cheap is wrong and scores low
    return sweep([0.9, 0.1], [True, False], [True, True], [1.0, 1.0], [10.0, 10.0])


def test_sweep_endpoints() -> None:
    points = toy()
    assert points[0] == {"frac_cheap": 0.0, "cost": 20.0, "accuracy": 1.0, "threshold": None}
    assert points[-1]["frac_cheap"] == 1.0
    assert points[-1]["cost"] == pytest.approx(2.0)
    assert points[-1]["accuracy"] == pytest.approx(0.5)


def test_sweep_routes_most_confident_first() -> None:
    middle = toy()[1]
    assert middle["cost"] == pytest.approx(11.0)
    assert middle["accuracy"] == pytest.approx(1.0)
    assert middle["threshold"] == pytest.approx(0.9)


def test_recommend_picks_cheapest_within_tolerance() -> None:
    assert recommend(toy())["cost"] == pytest.approx(11.0)

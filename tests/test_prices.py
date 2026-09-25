import pytest

from autopilot.prices import cost


def test_luna_price() -> None:
    assert cost("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_sol_price() -> None:
    assert cost("gpt-6-sol", 150, 300) == pytest.approx((150 * 2.00 + 300 * 10.00) / 1_000_000)


def test_dated_snapshot_is_priced() -> None:
    assert cost("gpt-6-luna-2026-09-22", 1_000_000, 0) == pytest.approx(0.10)


def test_unknown_model_is_unpriced() -> None:
    assert cost("gpt-4o", 10, 10) is None
    assert cost("gpt-6-lunar", 10, 10) is None

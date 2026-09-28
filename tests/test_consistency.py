import pytest

from autopilot.consistency import agreement, agreement_of_keys, answer_key


def test_key_is_the_answer_letter() -> None:
    assert answer_key("Let me think...\nAnswer: C") == "C"


def test_key_falls_back_to_normalised_last_line() -> None:
    assert answer_key("Reasoning here.\n\n  The capital is Paris. ") == "the capital is paris"


def test_key_of_empty_text_is_empty() -> None:
    assert answer_key("  \n ") == ""


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["Answer: A"] * 5, 1.0),
        (["Answer: A", "Answer: A", "Answer: A", "Answer: B", "Answer: C"], 0.6),
        (["Answer: A", "Answer: B", "Answer: C", "Answer: D", "Answer: E"], 0.2),
        (["", "", "", "", ""], 0.0),
        ([], 0.0),
    ],
)
def test_agreement(texts: list[str], expected: float) -> None:
    assert agreement(texts) == pytest.approx(expected)


def test_agreement_of_keys_treats_missing_first_key_as_unconfident() -> None:
    assert agreement_of_keys(["", "", ""]) == 0.0
    assert agreement_of_keys(["B", "B", "C", "B"]) == pytest.approx(0.75)

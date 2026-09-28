import pytest

from offline.grade import extract_letter, is_correct


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Step 1: ...\nAnswer: C", "C"),
        ("The answer is (C).", "C"),
        ("**Answer:** C", "C"),
        ("Final answer: J", "J"),
        ("C.", "C"),
        ("(J)", "J"),
        ("Answer: A\nWait, let me re-check.\nAnswer: D", "D"),
        ("The answer is a tricky one, no letter here", None),
        ("Answer: Both options look wrong", None),
        ("Answer: K", None),
        ("I think it is C", None),
        ("", None),
    ],
)
def test_extract_letter(text: str, expected: str | None) -> None:
    assert extract_letter(text) == expected


def test_is_correct() -> None:
    assert is_correct("Answer: B", "B")
    assert not is_correct("Answer: C", "B")
    assert not is_correct("", "B")


@pytest.mark.parametrize("text", ["Answer: I think B", "Answer: I think so", "Answer: A or B",
                                  "The answer: E. coli"])
def test_hedged_prose_is_not_a_letter(text: str) -> None:
    assert extract_letter(text) is None

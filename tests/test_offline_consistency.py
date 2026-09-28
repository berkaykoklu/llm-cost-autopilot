from offline.consistency import letter_confidence


def tok(token: str, logprob: float) -> dict[str, object]:
    return {"token": token, "logprob": logprob}


def test_confidence_of_the_last_matching_letter_token() -> None:
    content = [tok("B", -3.0), tok("Answer", 0.0), tok(":", 0.0), tok(" C", -0.1)]
    assert abs(letter_confidence(content, "C") - 0.9048) < 1e-3


def test_confidence_zero_without_letter_or_logprobs() -> None:
    assert letter_confidence([tok(" C", 0.0)], None) == 0.0
    assert letter_confidence([], "C") == 0.0

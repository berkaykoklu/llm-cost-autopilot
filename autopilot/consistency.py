import string

from offline.grade import extract_letter


def answer_key(text: str) -> str:
    """What two answers must share to count as the same answer."""
    letter = extract_letter(text)
    if letter:
        return letter
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip(string.punctuation + " ").lower() if lines else ""


def agreement_of_keys(keys: list[str]) -> float:
    """Share of samples that give the same answer as the first one (the one we would serve)."""
    if not keys or not keys[0]:
        return 0.0  # an empty or unreadable answer never counts as confident
    return sum(k == keys[0] for k in keys) / len(keys)


def agreement(texts: list[str]) -> float:
    return agreement_of_keys([answer_key(t) for t in texts])

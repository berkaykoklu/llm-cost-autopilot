import re

# "answer" in any case, then optional "is", colons, spaces or markdown stars, then an
# uppercase A-J not followed by a letter (so "answer is a ..." and "Answer: Both" don't match)
_ANSWER = re.compile(r"(?i:answer)(?:\s+is)?[\s*:]*\(?([A-J])\)?(?![A-Za-z])")
_BARE = re.compile(r"\(?([A-J])\)?\.?")


def extract_letter(text: str) -> str | None:
    found = _ANSWER.findall(text)
    if found:
        return str(found[-1])  # last one wins: models restate or correct themselves
    bare = _BARE.fullmatch(text.strip())
    return bare.group(1) if bare else None


def is_correct(text: str, gold: str) -> bool:
    return extract_letter(text) == gold

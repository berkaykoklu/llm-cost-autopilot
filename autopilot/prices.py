CHEAP = "gpt-6-luna"
EXPENSIVE = "gpt-6-sol"

# cheap answers fast; expensive thinks first. Routing decides whether thinking is worth paying for.
# Hidden reasoning tokens are billed as output.
REASONING_EFFORT = {CHEAP: "none", EXPENSIVE: "medium"}

# $ per 1M tokens (input, output), standard tier, verified 2026-09-25 on the OpenAI pricing page
PRICES = {
    CHEAP: (0.10, 0.50),
    EXPENSIVE: (2.00, 10.00),
}


import re


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    for name, (inp, out) in PRICES.items():
        # dated snapshots such as gpt-6-luna-2026-09-22 share the base price; gpt-6-luna-mini does not
        if re.fullmatch(rf"{name}(-\d{{4}}-\d{{2}}-\d{{2}})?", model):
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
    return None

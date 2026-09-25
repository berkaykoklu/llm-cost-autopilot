CHEAP = "gpt-6-luna"
EXPENSIVE = "gpt-6-sol"

# $ per 1M tokens (input, output), standard tier, verified 2026-09-25 on the OpenAI pricing page
PRICES = {
    CHEAP: (0.10, 0.50),
    EXPENSIVE: (2.00, 10.00),
}


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    for name, (inp, out) in PRICES.items():
        # dated snapshots such as gpt-6-luna-2026-09-22 share the base price
        if model == name or model.startswith(name + "-"):
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
    return None

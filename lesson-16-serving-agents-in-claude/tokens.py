"""Token estimation.

`estimate_tokens` is a HEURISTIC: about 4 characters per token for English prose. Code, numbers, and non-English text
usually tokenize into MORE tokens than this predicts, so budgets built on it should keep a safety margin. Everything in this
lesson takes a `count` function as a parameter, so you can plug in your provider's real token counter. I did not test any
provider's counting endpoint.
"""

import math


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0

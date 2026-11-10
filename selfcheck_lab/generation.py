"""Greedy generation that records, at every step, the chosen token's log-probability and its top alternatives.

This is what the OpenAI API returns when you pass `logprobs=True, top_logprobs=k`. Here we compute it
ourselves from the model's logits, so the module never imports a model: it receives a `next_logits`
function and works on plain numbers (which is what lets the tests use a fake model).
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class Alternative:
    token_id: int
    label: str          # token text with whitespace made visible
    logprob: float

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


@dataclass
class TokenStep:
    """One generated token: what was chosen, how likely it was, and what else the model considered."""
    index: int
    token_id: int
    text: str           # the exact text this token added to the output
    label: str          # same text with whitespace made visible (␣ for space, ⏎ for newline)
    logprob: float
    top: list[Alternative] = field(default_factory=list)   # top-k at this step, best first (includes the chosen token)

    @property
    def probability(self) -> float:
        """Token probability: e ** logprob (the notebook writes 2.718281828459045 ** logprob)."""
        return math.exp(self.logprob)

    @property
    def alternatives(self) -> list[Alternative]:
        """The top-k tokens the model did NOT choose."""
        return [a for a in self.top if a.token_id != self.token_id]


def greedy_generate(
    next_logits: Callable[[Optional[int]], torch.Tensor],
    eos_ids: set[int],
    max_new_tokens: int,
    top_k: int,
    decode: Callable[[list[int]], str],
    label: Callable[[int], str],
) -> list[TokenStep]:
    """Always pick the most likely token (temperature 0) and record its log-probability and the top-k.

    `next_logits(prev)` returns the logits for the next position; `prev` is None on the first call and
    the previously chosen token afterwards. The end-of-text token is not recorded as a step.
    """
    steps: list[TokenStep] = []
    ids: list[int] = []
    prev_text = ""
    prev: Optional[int] = None
    for i in range(max_new_tokens):
        logprobs = torch.log_softmax(next_logits(prev).float(), dim=-1)
        values, indices = torch.topk(logprobs, k=max(1, top_k))
        chosen = int(indices[0])
        if chosen in eos_ids:
            break
        ids.append(chosen)
        # Decode the whole prefix and take the difference, so characters split across tokens (é, ü, emoji)
        # are attributed to the token that completes them.
        # A token holding only part of a character decodes to "�"; it gets empty text and the next
        # token gets the whole character.
        text_now = decode(ids)
        if text_now.endswith("�"):
            piece = ""
        else:
            piece = text_now[len(prev_text):] if text_now.startswith(prev_text) else decode([chosen])
            prev_text = text_now
        top = [Alternative(int(t), label(int(t)), float(v)) for v, t in zip(values, indices)]
        steps.append(TokenStep(i, chosen, piece, label(chosen), float(logprobs[chosen]), top))
        prev = chosen
    return steps


def output_text(steps: list[TokenStep]) -> str:
    return "".join(s.text for s in steps)

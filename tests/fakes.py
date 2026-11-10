"""A tiny fake 'model': a script of next-token distributions, so tests need no download."""

import math

import torch

from selfcheck_lab.generation import Alternative, TokenStep

VOCAB = ["<eos>", "A", "B", "C", " ", "\n", "1", "2", ",", "."]
EOS = 0


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


def label(i):
    return VOCAB[i].replace(" ", "␣").replace("\n", "⏎")


def scripted(distributions):
    """next_logits that returns log(p) for each {token: p} dict in turn, ignoring the input."""
    it = iter(distributions)

    def next_logits(prev):
        probs = next(it)
        logits = torch.full((len(VOCAB),), -1e9)
        for tok, p in probs.items():
            logits[VOCAB.index(tok)] = math.log(p)
        return logits

    return next_logits


def step(text, p, index=0):
    """A TokenStep with a given text and probability (for tests that don't need generation)."""
    return TokenStep(index, 1, text, text, math.log(p), [Alternative(1, text, math.log(p))])


def steps_from(pairs):
    return [step(t, p, i) for i, (t, p) in enumerate(pairs)]

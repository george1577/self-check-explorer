"""Turn token probabilities into self-check signals (notebook 1's analyze_token_confidence and
calculate_response_confidence), with the notebook's hard-coded numbers made into parameters."""

from dataclasses import dataclass

from .generation import TokenStep

DEFAULT_WEIGHT = 0.7            # notebook: 0.7 * average + 0.3 * minimum
DEFAULT_LOW_FRACTION = 0.5      # notebook: a token is "low confidence" below 50% of the average


@dataclass
class ResponseConfidence:
    average: float              # mean token probability
    minimum: float              # lowest token probability
    maximum: float
    weight: float               # how much the average counts; the minimum gets (1 - weight)
    score: float                # response confidence score
    low_threshold: float        # low_fraction * average
    low_tokens: list[TokenStep] # tokens below low_threshold, in output order
    weakest: TokenStep          # the token that set the minimum


def response_confidence(steps: list[TokenStep], weight: float = DEFAULT_WEIGHT,
                        low_fraction: float = DEFAULT_LOW_FRACTION) -> ResponseConfidence:
    if not steps:
        raise ValueError("an empty response has no token probabilities")
    probs = [s.probability for s in steps]
    average = sum(probs) / len(probs)
    minimum = min(probs)
    score = round(weight * average + (1 - weight) * minimum, 4)
    threshold = low_fraction * average
    low = [s for s in steps if s.probability < threshold]
    weakest = min(steps, key=lambda s: s.probability)
    return ResponseConfidence(average, minimum, max(probs), weight, score, threshold, low, weakest)

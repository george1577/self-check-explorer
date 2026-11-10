# Self-Check Explorer

A learning tool for seeing how the Self-Check pattern uses a model's own token probabilities to flag answers, or parts of answers, that are likely to be wrong.

## Language

### Model output

**Log-probability**:
The natural log of the probability the model assigned to one generated token, as reported at generation time.
_Avoid_: Logprob score, confidence

**Token probability**:
The probability the model assigned to one chosen token, obtained as e raised to its log-probability.
_Avoid_: Confidence, certainty

**Alternative**:
A token the model ranked among its top choices at a step but did not choose.
_Avoid_: Candidate, option

### Self-check signals

**Line confidence**:
The lowest token probability among the tokens of one output line (one data row).
_Avoid_: Row score, confidence

**Response confidence score**:
A single number for a whole response: a weighted mix of the average and the minimum token probability.
_Avoid_: Confidence, certainty score

**Low-confidence token**:
A chosen token whose token probability falls below a set fraction of the response's average token probability.
_Avoid_: Uncertain token, weak token

### Receipts task

**Imputation**:
The model filling in a missing field of a receipt row from the fields that are present.
_Avoid_: Completion, prediction

**Verification error**:
The absolute difference between a row's paid amount and what its other fields imply, computed by ordinary arithmetic outside the model.
_Avoid_: Loss, accuracy

**Gap from correct**:
The largest difference, over all four fields, between the model's row and the row the rules actually produce.
_Avoid_: Error (ambiguous with verification error), accuracy

**Review threshold**:
The line confidence below which a row is sent for checking instead of being accepted automatically.
_Avoid_: Cutoff, confidence threshold (ambiguous with the low-confidence fraction)

**Self-check**:
The pattern of using a model's own token probabilities to decide which outputs to trust and which to send for verification.
_Avoid_: Self-consistency, self-reflection

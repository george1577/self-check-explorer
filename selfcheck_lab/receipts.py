"""Notebook 2: the model fills in missing receipt fields, and each output line gets a line confidence.

Everything here is plain arithmetic on token probabilities and numbers, so it can be tested exactly.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from .generation import TokenStep

TAX_RATE = 0.0921
FIELDS = ("billed_amount", "tax", "tip", "paid_amount")

# The notebook's six rows, verbatim (including the trailing space on row 4).
NOTEBOOK_ROWS = """312.32,28.76,60,401.08
312.32,28.76,,400
312.32,28.76,60,
312.21,,50,
312.43,,,400
300,27.63,60,387.63"""


@dataclass
class ReceiptRow:
    billed_amount: Optional[float]
    tax: Optional[float]
    tip: Optional[float]
    paid_amount: Optional[float]

    def values(self) -> tuple:
        return (self.billed_amount, self.tax, self.tip, self.paid_amount)

    def missing(self) -> list[str]:
        return [name for name, v in zip(FIELDS, self.values()) if v is None]

    def as_csv(self) -> str:
        return ",".join("" if v is None else _money(v) for v in self.values())


def _money(v: float) -> str:
    """312.3 -> '312.3', 60.0 -> '60', like the notebook's input."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _number(text: str) -> Optional[float]:
    text = text.strip().replace("$", "")
    return float(text) if text else None


def parse_row(line: str) -> Optional[ReceiptRow]:
    """'312.21,,50,' -> ReceiptRow(312.21, None, 50.0, None). None if the line isn't 4 numbers-or-blanks."""
    parts = line.split(",")
    if len(parts) != 4:
        return None
    try:
        return ReceiptRow(*[_number(p) for p in parts])
    except ValueError:
        return None


def parse_input(text: str) -> list[ReceiptRow]:
    """Every non-empty line of the user's input. Raises ValueError naming the first bad line."""
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = parse_row(line)
        if row is None:
            raise ValueError(f"line {n} is not 4 comma-separated values: {line!r}")
        rows.append(row)
    return rows


def correct_row(row: ReceiptRow) -> Optional[ReceiptRow]:
    """Fill the missing fields with the system prompt's own rules, using exact arithmetic.

    Returns None when the rules can't determine the row (billed amount missing, or tip and paid both missing).
    """
    billed, tax, tip, paid = row.values()
    if billed is None:
        return None
    if tax is None:
        tax = round(billed * TAX_RATE, 2)
    if tip is None and paid is None:
        return None
    if tip is None:
        tip = round(paid - billed - tax, 2)
    if paid is None:
        paid = round(billed + tax + tip, 2)
    return ReceiptRow(billed, tax, tip, paid)


def random_rows(n: int, seed: int = 0, complete_share: float = 0.5) -> list[ReceiptRow]:
    """n consistent receipts, reproducible from seed. About `complete_share` of them are left complete
    (nothing to fill in, like most real receipts); each other row has exactly one of tax / tip / paid blanked."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        billed = round(rng.uniform(15, 400), 2)
        tax = round(billed * TAX_RATE, 2)
        tip = rng.choice([round(billed * rng.choice([0.15, 0.18, 0.20]), 2), float(rng.randrange(5, 80, 5))])
        paid = round(billed + tax + tip, 2)
        values = [billed, tax, tip, paid]
        if rng.random() >= complete_share:
            values[rng.choice([1, 2, 3])] = None
        rows.append(ReceiptRow(*values))
    return rows


# ---- splitting the output into lines (the notebook's parse_result loop) -----------------

@dataclass
class OutputLine:
    tokens: list[TokenStep]

    @property
    def text(self) -> str:
        return "".join(t.text for t in self.tokens)

    @property
    def line_confidence(self) -> float:
        """The lowest token probability in the line (the notebook's min over the line's tokens)."""
        return min(t.probability for t in self.tokens)

    @property
    def weakest(self) -> TokenStep:
        return min(self.tokens, key=lambda t: t.probability)


def split_lines(steps: list[TokenStep]) -> tuple[list[OutputLine], list[OutputLine]]:
    """Group tokens into lines exactly as the notebook does: a token containing a newline ends its line
    (so the newline token counts toward the line it ends).

    Returns (data_lines, skipped_lines). Lines with no digit at all (blank lines, ``` fences, chatter)
    are skipped so they don't shift the alignment with the input rows. The notebook has no such step;
    GPT never emitted those lines.
    """
    lines, current = [], []
    for step in steps:
        current.append(step)
        if "\n" in step.text:
            lines.append(OutputLine(current))
            current = []
    if current:
        lines.append(OutputLine(current))
    data = [l for l in lines if any(c.isdigit() for c in l.text)]
    skipped = [l for l in lines if not any(c.isdigit() for c in l.text)]
    return data, skipped


# ---- checking each line -----------------------------------------------------------------

@dataclass
class LineResult:
    row_no: int                         # 0-based input row
    given: Optional[ReceiptRow]         # the input row (None for extra output lines)
    line: Optional[OutputLine]          # the model's line (None if it produced too few lines)
    output: Optional[ReceiptRow]        # the model's line parsed (None if unparseable / missing)
    correct: Optional[ReceiptRow]       # exact answer from the prompt's rules (None if undetermined)

    @property
    def status(self) -> str:
        if self.given is None:
            return "extra line"
        if self.line is None:
            return "no output"
        if self.output is None or None in self.output.values():
            return "unparseable"
        return "ok"

    @property
    def parsed(self) -> bool:
        return self.status == "ok"

    @property
    def line_confidence(self) -> Optional[float]:
        return self.line.line_confidence if self.line else None

    @property
    def changed_given(self) -> list[str]:
        """Fields the input already had but the model wrote differently."""
        if not self.parsed:
            return []
        return [name for name, g, o in zip(FIELDS, self.given.values(), self.output.values())
                if g is not None and abs(g - o) > 0.005]

    @property
    def truth_gap(self) -> Optional[float]:
        """Largest absolute difference between the model's row and the correct row, over all four fields."""
        if not self.parsed or self.correct is None:
            return None
        return round(max(abs(o - c) for o, c in zip(self.output.values(), self.correct.values())), 2)

    @property
    def verification_error(self) -> Optional[float]:
        """|billed + tax + tip - paid| using the model's own tax value: does the row add up?"""
        if not self.parsed:
            return None
        b, tax, tip, paid = self.output.values()
        return round(abs(b + tax + tip - paid), 2)

    @property
    def notebook_verification_error(self) -> Optional[float]:
        """The notebook's verify_row: |round(billed * 1.0921, 2) + tip - paid|, ignoring the tax column."""
        if not self.parsed:
            return None
        b, _, tip, paid = self.output.values()
        return round(abs(round(b * (1 + TAX_RATE), 2) + tip - paid), 2)

    def is_wrong(self, tolerance: float) -> Optional[bool]:
        """True if off from the correct row by more than tolerance. Unparseable rows count as wrong.
        None when the correct answer can't be determined."""
        if self.given is None:
            return None
        if not self.parsed:
            return True
        if self.truth_gap is None:
            return None
        return self.truth_gap > tolerance + 1e-9


def check_lines(given: list[ReceiptRow], steps: list[TokenStep]) -> tuple[list[LineResult], list[OutputLine]]:
    """Pair each input row with its output line (by billed amount, in order) and check it.
    Output lines no row claimed are returned as 'extra line' results at the end.
    Returns (results, skipped_lines)."""
    data, skipped = split_lines(steps)
    results, used, j = [], set(), 0
    for i, g in enumerate(given):
        # Walk forward through the output to the next line that starts with this row's billed amount
        # (the model always copies it). Matching by position alone would misalign every row after a
        # skipped or merged line and compare the model's answers against the wrong receipts.
        k = next((k for k in range(j, len(data)) if _billed_matches(data[k], g.billed_amount)), None)
        if k is None and g.billed_amount is None and j < len(data):
            k = j
        line = data[k] if k is not None else None
        if k is not None:
            used.add(k)
            j = k + 1
        output = parse_row(line.text.strip()) if line else None
        results.append(LineResult(i, g, line, output, correct_row(g)))
    for k, line in enumerate(data):
        if k not in used:
            results.append(LineResult(len(results), None, line, parse_row(line.text.strip()), None))
    return results, skipped


def _billed_matches(line: OutputLine, billed: Optional[float]) -> bool:
    if billed is None:
        return False
    try:
        return abs(float(line.text.split(",")[0].strip().replace("$", "")) - billed) < 0.005
    except ValueError:
        return False


# ---- acting on the signal ------------------------------------------------------------------

@dataclass
class Triage:
    threshold: float
    tolerance: float
    caught: list[int] = field(default_factory=list)        # wrong, sent to review   (good)
    false_alarms: list[int] = field(default_factory=list)  # right, sent to review   (wasted effort)
    missed: list[int] = field(default_factory=list)        # wrong, auto-accepted    (bad)
    accepted: list[int] = field(default_factory=list)      # right, auto-accepted    (good)
    unknown: list[int] = field(default_factory=list)       # correct answer undetermined


def needs_review(result: LineResult, threshold: float) -> bool:
    """Send to review if the line confidence is below threshold, or the line couldn't be read at all."""
    return not result.parsed or result.line_confidence < threshold


def triage(results: list[LineResult], threshold: float, tolerance: float = 0.01) -> Triage:
    t = Triage(threshold, tolerance)
    for r in results:
        if r.given is None:
            continue
        wrong = r.is_wrong(tolerance)
        review = needs_review(r, threshold)
        if wrong is None:
            t.unknown.append(r.row_no)
        elif wrong and review:
            t.caught.append(r.row_no)
        elif wrong:
            t.missed.append(r.row_no)
        elif review:
            t.false_alarms.append(r.row_no)
        else:
            t.accepted.append(r.row_no)
    return t

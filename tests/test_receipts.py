import math

from selfcheck_lab.receipts import (
    NOTEBOOK_ROWS, ReceiptRow, check_lines, correct_row, parse_input, parse_row, random_rows, split_lines, triage,
)
from tests.fakes import steps_from


def test_parse_row_handles_blanks_and_spaces():
    assert parse_row("312.21,,50, ") == ReceiptRow(312.21, None, 50.0, None)
    assert parse_row("1,2,3") is None
    assert parse_row("a,b,c,d") is None


def test_notebook_rows_parse():
    rows = parse_input(NOTEBOOK_ROWS)
    assert len(rows) == 6
    assert rows[4].missing() == ["tax", "tip"]


def test_correct_row_follows_prompt_rules():
    rows = parse_input(NOTEBOOK_ROWS)
    assert correct_row(rows[1]) == ReceiptRow(312.32, 28.76, 58.92, 400.0)     # tip = 400 - 312.32 - 28.76
    assert correct_row(rows[2]) == ReceiptRow(312.32, 28.76, 60.0, 401.08)     # paid = sum
    assert correct_row(rows[3]) == ReceiptRow(312.21, 28.75, 50.0, 390.96)     # tax first, then paid
    assert correct_row(rows[4]) == ReceiptRow(312.43, 28.77, 58.8, 400.0)      # tax, then tip
    assert correct_row(ReceiptRow(10.0, None, None, None)) is None


def test_random_rows_are_reproducible_and_blank_at_most_one_field():
    a, b = random_rows(10, seed=3), random_rows(10, seed=3)
    assert a == b
    assert all(len(r.missing()) <= 1 and r.billed_amount is not None for r in a)
    assert all(len(r.missing()) == 1 for r in random_rows(10, seed=3, complete_share=0.0))
    assert all(not r.missing() for r in random_rows(10, seed=3, complete_share=1.0))
    for r in a:
        full = correct_row(r)
        assert math.isclose(full.billed_amount + full.tax + full.tip, full.paid_amount, abs_tol=0.011)


def test_csv_round_trip():
    assert ReceiptRow(300.0, 27.63, 60.0, None).as_csv() == "300,27.63,60,"


def test_split_lines_newline_token_belongs_to_line_it_ends():
    steps = steps_from([("1", 0.9), (",", 0.5), ("\n", 0.3), ("```", 0.8), ("\n", 0.9), ("2", 0.7)])
    data, skipped = split_lines(steps)
    assert [l.text for l in data] == ["1,\n", "2"]
    assert math.isclose(data[0].line_confidence, 0.3)       # the newline (0.3) is the line's minimum
    assert [l.text for l in skipped] == ["```\n"]


def _line(text, p_min):
    """Tokens for one output line: every character at 0.99 except the first digit at p_min."""
    pairs = [(c, 0.99) for c in text]
    pairs[0] = (text[0], p_min)
    return pairs


def test_check_lines_and_triage():
    rows = [ReceiptRow(100.0, None, 10.0, 119.21), ReceiptRow(100.0, 9.21, None, 120.0)]
    # row 0 correct (tax 9.21) but low confidence; row 1 wrong tip (should be 10.79) but confident
    steps = steps_from(_line("100,9.21,10,119.21\n", 0.4) + _line("100,9.21,11,120", 0.95))
    results, _ = check_lines(rows, steps)
    assert [r.status for r in results] == ["ok", "ok"]
    assert results[0].truth_gap == 0.0
    assert results[1].truth_gap == 0.21
    assert results[1].verification_error == 0.21
    t = triage(results, threshold=0.5, tolerance=0.01)
    assert t.false_alarms == [0] and t.missed == [1]


def test_changed_given_and_missing_output():
    rows = [ReceiptRow(312.43, None, None, 400.0), ReceiptRow(1.0, 1.0, 1.0, None)]
    steps = steps_from([(c, 0.9) for c in "312.43,28.80,60,401.23"])
    results, _ = check_lines(rows, steps)
    assert results[0].changed_given == ["paid_amount"]
    assert results[1].status == "no output"
    assert results[1].is_wrong(0.01) is True


def test_notebook_verification_ignores_model_tax():
    rows = [ReceiptRow(312.21, None, 50.0, None)]
    steps = steps_from([(c, 0.9) for c in "312.21,28.84,50,391.05"])   # GPT's actual row 4
    (r,), _ = check_lines(rows, steps)
    assert r.verification_error == 0.0          # adds up with its own (wrong) tax
    assert r.notebook_verification_error == 0.09
    assert r.truth_gap == 0.09


def test_rows_are_matched_by_billed_amount_when_model_skips_a_line():
    rows = [ReceiptRow(10.0, 1.0, 1.0, 12.0), ReceiptRow(20.0, 2.0, 2.0, None), ReceiptRow(30.0, 3.0, 3.0, 36.0)]
    steps = steps_from([(c, 0.9) for c in "10,1,1,12\n30,3,3,36\n99,1,1,101"])   # skipped row 1, added junk
    results, _ = check_lines(rows, steps)
    assert [r.status for r in results] == ["ok", "no output", "ok", "extra line"]
    assert results[2].truth_gap == 0.0

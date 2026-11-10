import math

from selfcheck_lab.generation import greedy_generate, output_text
from tests.fakes import EOS, decode, label, scripted


def test_records_chosen_token_probability_and_alternatives():
    steps = greedy_generate(
        scripted([{"A": 0.6, "B": 0.3, "C": 0.1}, {"B": 0.9, "C": 0.1}, {"<eos>": 1.0}]),
        {EOS}, max_new_tokens=10, top_k=3, decode=decode, label=label,
    )
    assert output_text(steps) == "AB"
    assert math.isclose(steps[0].probability, 0.6, rel_tol=1e-6)
    assert [a.label for a in steps[0].alternatives] == ["B", "C"]
    assert math.isclose(steps[0].alternatives[0].probability, 0.3, rel_tol=1e-6)
    assert math.isclose(steps[1].probability, 0.9, rel_tol=1e-6)


def test_stops_at_max_new_tokens():
    steps = greedy_generate(scripted([{"A": 1.0}] * 5), {EOS}, 3, 2, decode, label)
    assert len(steps) == 3

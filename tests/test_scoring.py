import math

import pytest

from selfcheck_lab.scoring import response_confidence
from tests.fakes import steps_from


def test_matches_notebook_formula():
    # probabilities 1.0, 0.9, 0.2 -> avg 0.7, min 0.2 -> 0.7*0.7 + 0.3*0.2 = 0.55
    rc = response_confidence(steps_from([("A", 1.0), ("B", 0.9), ("C", 0.2)]))
    assert math.isclose(rc.average, 0.7)
    assert math.isclose(rc.minimum, 0.2)
    assert rc.score == 0.55
    assert rc.weakest.text == "C"


def test_low_tokens_are_below_fraction_of_average():
    # avg 0.7 -> threshold 0.35; only C (0.2) is below
    rc = response_confidence(steps_from([("A", 1.0), ("B", 0.9), ("C", 0.2)]), low_fraction=0.5)
    assert [s.text for s in rc.low_tokens] == ["C"]
    assert math.isclose(rc.low_threshold, 0.35)


def test_weight_one_is_plain_average():
    rc = response_confidence(steps_from([("A", 1.0), ("B", 0.5)]), weight=1.0)
    assert rc.score == 0.75


def test_empty_response_rejected():
    with pytest.raises(ValueError):
        response_confidence([])

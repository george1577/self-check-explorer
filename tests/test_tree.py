from selfcheck_lab.tree import token_tree_dot
from tests.fakes import steps_from


def test_tree_chains_chosen_tokens_and_truncates():
    dot = token_tree_dot(steps_from([("A", 0.9), ("B", 0.2), ("C", 0.8)]), max_tokens=2, low_threshold=0.3)
    assert "root -> t0" in dot and "t0 -> t1" in dot
    assert "t2" not in dot and "1 more tokens" in dot
    assert dot.count("penwidth=3") == 1     # only B is below the threshold

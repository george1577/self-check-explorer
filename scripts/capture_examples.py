"""Print the numbers quoted in docs/how-self-check-works.md, so they can be reproduced.

    uv run python scripts/capture_examples.py questions 0      # notebook 1 questions, 0.5B model
    uv run python scripts/capture_examples.py steps 0 "John"   # every token of one preset (name prefix)
    uv run python scripts/capture_examples.py receipts 0       # notebook 2 rows, 0.5B model
    uv run python scripts/capture_examples.py receipts 0 12 1  # 12 random rows, seed 1, 0.5B model
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selfcheck_lab.lm import MODELS, LanguageModel  # noqa: E402
from selfcheck_lab.pipelines import answer_question, impute_receipts  # noqa: E402
from selfcheck_lab.prompts import QUESTIONS  # noqa: E402
from selfcheck_lab.receipts import NOTEBOOK_ROWS, parse_input, random_rows, triage  # noqa: E402
from selfcheck_lab.scoring import response_confidence  # noqa: E402


def alts(s, n=3):
    return [(a.label, round(a.probability, 4)) for a in s.alternatives[:n]]


def questions(lm):
    for name, q in QUESTIONS.items():
        t = time.time()
        run = answer_question(lm, q, 150, 5)
        rc = response_confidence(run.steps)
        print(f"\n### {name}  ({time.time() - t:.1f}s, {len(run.steps)} tokens)")
        print("ANSWER:", run.answer)
        print(f"avg={rc.average:.4f} min={rc.minimum:.4f} score={rc.score} weakest={rc.weakest.label!r} "
              f"threshold={rc.low_threshold:.4f}")
        print("LOW:", [(s.index, s.label, round(s.probability, 4), alts(s)) for s in rc.low_tokens])


def steps(lm, prefix):
    name = next(n for n in QUESTIONS if n.startswith(prefix))
    run = answer_question(lm, QUESTIONS[name], 150, 5)
    print(run.answer)
    for s in run.steps:
        print(s.index, repr(s.label), round(s.probability, 4), alts(s, 4))


def receipts(lm, n, seed=0):
    rows = random_rows(n, seed=seed) if n else parse_input(NOTEBOOK_ROWS)
    t = time.time()
    run = impute_receipts(lm, rows)
    print(f"INPUT ({time.time() - t:.1f}s):\n{run.input_csv}OUTPUT:\n{run.output}\n")
    for r in run.results:
        weak = r.line.weakest if r.line else None
        print(r.row_no, r.status, r.given.as_csv() if r.given else None, "->",
              r.line.text.strip() if r.line else None,
              "| correct", r.correct.as_csv() if r.correct else None,
              f"| conf={r.line_confidence:.4f}" if r.line else "",
              f"weakest={weak.label!r}@{weak.index} alts={alts(weak)}" if weak else "",
              "| gap", r.truth_gap, "| verr", r.verification_error, "| nb_verr", r.notebook_verification_error,
              "| changed", r.changed_given)
        if r.line and not n:
            print("    tokens:", [(s.label, round(s.probability, 3)) for s in r.line.tokens])
    for th in (0.3, 0.5, 0.7, 0.9):
        tr = triage(run.results, th, 0.01)
        print(f"threshold {th}: caught={tr.caught} false_alarms={tr.false_alarms} missed={tr.missed} "
              f"accepted={tr.accepted} unknown={tr.unknown}")


if __name__ == "__main__":
    mode, model = sys.argv[1], LanguageModel(MODELS[int(sys.argv[2])])
    if mode == "questions":
        questions(model)
    elif mode == "steps":
        steps(model, sys.argv[3])
    else:
        receipts(model, int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 0)

"""Wire the language model to each tab. The Streamlit app calls only these functions."""

from dataclasses import dataclass

from .generation import TokenStep, greedy_generate, output_text
from .lm import LanguageModel
from .prompts import question_messages, receipt_messages
from .receipts import LineResult, OutputLine, ReceiptRow, check_lines


@dataclass
class AnswerRun:
    question: str
    answer: str
    steps: list[TokenStep]
    model_id: str


def answer_question(lm: LanguageModel, question: str, max_new_tokens: int = 150, top_k: int = 5) -> AnswerRun:
    """Tabs ① and ②: greedy answer with the token probability and top-k alternatives at every step."""
    prompt = lm.chat_prompt_ids(question_messages(question))
    steps = greedy_generate(lm.stepper(prompt), lm.eos_ids, max_new_tokens, top_k, lm.decode, lm.token_label)
    return AnswerRun(question, output_text(steps), steps, lm.model_id)


@dataclass
class ReceiptRun:
    rows: list[ReceiptRow]
    input_csv: str
    output: str
    steps: list[TokenStep]
    results: list[LineResult]
    skipped: list[OutputLine]
    model_id: str


def impute_receipts(lm: LanguageModel, rows: list[ReceiptRow], top_k: int = 5) -> ReceiptRun:
    """Tab ③: one batched call for all rows (as in the notebook), then a line confidence per row."""
    input_csv = "\n".join(r.as_csv() for r in rows) + "\n"
    prompt = lm.chat_prompt_ids(receipt_messages(input_csv))
    max_new_tokens = 30 * len(rows) + 40
    steps = greedy_generate(lm.stepper(prompt), lm.eos_ids, max_new_tokens, top_k, lm.decode, lm.token_label)
    results, skipped = check_lines(rows, steps)
    return ReceiptRun(rows, input_csv, output_text(steps), steps, results, skipped, lm.model_id)

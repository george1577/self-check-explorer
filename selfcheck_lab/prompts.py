"""Prompts copied from the notebooks."""

# Notebook 1's questions, in the order the notebook asks them.
QUESTIONS = {
    "Ataturk (well-known fact)": "What year was Ataturk born? Answer in one sentence.",
    "Clara Schumann (well-known fact)": "Who is Clara Schumann? Answer in one sentence.",
    "John Cole Howard (made-up person)": "Who is John Cole Howard? Answer in one sentence.",
    "Marie Curie's lab conditions (unknowable detail)": (
        "What was the exact time and temperature when Marie Curie made her first radium discovery? "
        "Include the barometric pressure in the lab."
    ),
    "Tesla & Edison joint patent (false premise)": (
        "Describe the collaboration between Nikola Tesla and Thomas Edison on their joint patent for "
        "wireless energy transmission in 1891. What were the specific technical details?"
    ),
}

QUESTION_SYSTEM = "You are a helpful AI assistant."

# Notebook 2's system prompt, with the notebook's indentation removed.
RECEIPT_SYSTEM = """You are a helpful AI assistant that helps parse restaurant receipts.
I will give you a set of parsed values containing on each line:

billed_amount, tax, tip, paid_amount

If tax is missing, calculate it as 9.21% of the billed_amount
If the tip is missing, calculate it as (paid_amount - billed_amount - tax)
If the paid_amount is missing, calculate it as (billed_amount + tax + tip)
Do not add any headers or explanations."""

# Small local models ignore "no explanations" and write prose, so the app appends an explicit format
# instruction with one worked line. GPT didn't need this; the guide lists it as a difference.
RECEIPT_FORMAT = """
Reply with exactly one line per input line, in the same order. Each line is the four numbers
billed_amount,tax,tip,paid_amount separated by commas, and every line ends with a line break.

Example input:
100,,10,119.21
50,4.61,,64.61

Example reply:
100,9.21,10,119.21
50,4.61,10,64.61"""


def question_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": QUESTION_SYSTEM}, {"role": "user", "content": question}]


def receipt_messages(rows_csv: str) -> list[dict]:
    return [{"role": "system", "content": RECEIPT_SYSTEM + RECEIPT_FORMAT}, {"role": "user", "content": rows_csv}]

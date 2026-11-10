"""Self-Check Explorer: a Streamlit app for seeing how token probabilities flag likely errors.

Run with:  uv run streamlit run app.py
"""

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from selfcheck_lab.generation import TokenStep
from selfcheck_lab.lm import MODELS, LanguageModel
from selfcheck_lab.pipelines import answer_question, impute_receipts
from selfcheck_lab.prompts import QUESTIONS
from selfcheck_lab.receipts import FIELDS, NOTEBOOK_ROWS, ReceiptRow, parse_input, random_rows, triage
from selfcheck_lab.scoring import DEFAULT_LOW_FRACTION, DEFAULT_WEIGHT, response_confidence
from selfcheck_lab.tree import probability_color, token_tree_dot

DOC_LINK = "docs/how-self-check-works.md"
CUSTOM = "✏️ Custom question…"

st.set_page_config(page_title="Self-Check Explorer", page_icon="🔎", layout="wide")

# Results in session_state are selfcheck_lab objects. When that code changes while the app runs, old
# results keep their old shape and could crash a tab, so drop them whenever the code changes.
RESULT_KEYS = ("t1", "t2_runs", "t3")


def code_version() -> str:
    files = sorted((Path(__file__).parent / "selfcheck_lab").glob("*.py"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


if st.session_state.get("_code_version") != (version := code_version()):
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_code_version"] = version


# max_entries=1: switching models unloads the previous one, so an 8 GB laptop never holds both.
@st.cache_resource(show_spinner="Loading model (first run downloads it)…", max_entries=1)
def get_model(model_id: str) -> LanguageModel:
    return LanguageModel(model_id)


# ---- shared display helpers ----------------------------------------------------------------

def alternatives_text(step: TokenStep, n: int = 5) -> str:
    return ", ".join(f"{a.label} {a.probability:.0%}" for a in step.alternatives[:n])


def heatmap_html(steps: list[TokenStep], low_threshold: float | None = None, marked: set[int] = frozenset()) -> str:
    """The output text with every token's background coloured by its token probability (red = low).
    Tokens below low_threshold get a red outline; tokens in `marked` get a thick blue underline.
    Hover a token to see its probability and alternatives."""
    spans = []
    for s in steps:
        style = f"background:{probability_color(s.probability)};border-radius:3px;padding:1px 0;"
        if low_threshold is not None and s.probability < low_threshold:
            style += "outline:2px solid #c00000;"
        if s.index in marked:
            style += "border-bottom:4px solid #1f4fd1;"
        tip = f"#{s.index}  '{s.label}'  p = {s.probability:.1%}"
        if s.alternatives:
            tip += "  |  alternatives: " + alternatives_text(s)
        text = html.escape(s.text).replace("\n", "⏎<br>")
        spans.append(f'<span title="{html.escape(tip)}" style="{style}">{text}</span>')
    return ('<div style="font-family:monospace;font-size:15px;line-height:2.1;white-space:pre-wrap;color:#111;">'
            + "".join(spans) + "</div>")


def legend():
    st.caption("Colour = token probability: 🟥 low · 🟧 middling · 🟩 high. Hover any token for its exact "
               "probability and the alternatives the model considered.")


def step_table(steps: list[TokenStep], low_threshold: float | None = None) -> pd.DataFrame:
    rows = []
    for s in steps:
        row = {
            "step": s.index,
            "token": s.label,
            "token probability": s.probability,
            "log-probability": round(s.logprob, 4),
            "top alternatives (not chosen)": alternatives_text(s),
        }
        if low_threshold is not None:
            row["low-confidence?"] = "⚠️ yes" if s.probability < low_threshold else ""
        rows.append(row)
    return pd.DataFrame(rows)


PROB_COLUMN = st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)


# ---- sidebar ---------------------------------------------------------------------------------

with st.sidebar:
    st.header("🔎 Self-Check Explorer")
    model_id = st.selectbox("Model for tabs ① and ②", MODELS, index=0,
                            help="0.5B is fast and hallucinates readily, which makes the pattern easy to see.")
    st.markdown(
        "Self-check: use the model's **own token probabilities** to spot answers, or parts of answers, "
        "it was unsure about, and send those for checking.\n\n"
        f"New to this? Read **[the guide]({DOC_LINK})** first. Terms are defined in `GLOSSARY.md`."
    )
    st.caption("Generation is greedy (temperature 0), so the same input always gives the same output.")

st.title("Self-Check Explorer")
tab1, tab2, tab3 = st.tabs(["① Token confidence", "② Response confidence score", "③ Self-check on receipts"])


# =============================================================================================
# ① Token confidence
# =============================================================================================
with tab1:
    st.markdown(
        "Ask a question and see, for **every token** of the answer, how likely the model thought it was "
        "and what else it almost wrote. This is notebook 1's `analyze_logprobs` graph."
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        preset = st.selectbox("Question", list(QUESTIONS) + [CUSTOM], index=2, key="t1_preset")
        question = (st.text_input("Your question", "Who is Mary Ellen Thornbury? Answer in one sentence.",
                                  key="t1_custom") if preset == CUSTOM else QUESTIONS[preset])
        st.markdown(f"> {question}")
    with c2:
        top_k = st.slider("Top-k alternatives per step", 1, 10, 5, key="t1_k",
                          help="The notebook asks the API for top_logprobs=5.")
        max_new = st.slider("Max answer length (tokens)", 20, 300, 150, key="t1_len")
        tree_len = st.slider("Tokens to draw in the tree", 5, 80, 30, key="t1_tree",
                             help="Long answers make a very wide tree; the table below always has every token.")
    if st.button("Ask the model", type="primary", key="t1_go"):
        with st.spinner("Generating…"):
            st.session_state["t1"] = answer_question(get_model(model_id), question, max_new, top_k)

    run = st.session_state.get("t1")
    if run:
        st.subheader("The answer")
        st.markdown(f"*{run.model_id}* answered:")
        st.info(run.answer)
        st.subheader("The token tree")
        st.caption(
            "Read left to right. Each **coloured box** is the token the model chose, with its probability. "
            "The **grey dashed boxes** branching from the same parent are the alternatives it ranked next, "
            "the roads not taken. A chosen token at 95% means the model barely hesitated; one at 15% means "
            "several continuations were nearly as likely."
        )
        st.graphviz_chart(token_tree_dot(run.steps, tree_len), width="stretch")
        st.subheader("Every step")
        st.caption("token probability = e^(log-probability). ␣ is a space, ⏎ a line break.")
        st.dataframe(step_table(run.steps), column_config={"token probability": PROB_COLUMN},
                     hide_index=True, width="stretch")
        st.markdown("**Look at one step closely**")
        i = st.slider("Step", 0, len(run.steps) - 1,
                      int(min(run.steps, key=lambda s: s.probability).index), key="t1_step",
                      help="Starts at the least likely token in the answer.")
        s = run.steps[i]
        so_far = "".join(x.text for x in run.steps[:i])
        st.markdown(f"Text so far: `…{html.escape(so_far[-80:])}` → next token **`{s.label}`** "
                    f"chosen with p = **{s.probability:.1%}**")
        chart = pd.DataFrame({"token": [a.label for a in s.top], "probability": [a.probability for a in s.top],
                              "chosen": ["chosen" if a.token_id == s.token_id else "alternative" for a in s.top]})
        st.altair_chart(
            alt.Chart(chart).mark_bar().encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("token:N", sort="-x"),
                color=alt.Color("chosen:N", scale=alt.Scale(domain=["chosen", "alternative"],
                                                            range=["#2e7d32", "#bbbbbb"])),
                tooltip=["token", alt.Tooltip("probability:Q", format=".2%")],
            ).properties(height=40 * len(s.top) + 30),
            width="stretch",
        )
        st.caption(f"The top {len(s.top)} tokens hold {sum(a.probability for a in s.top):.1%} of the probability; "
                   "the rest is spread across the other ~150,000 tokens in the vocabulary.")


# =============================================================================================
# ② Response confidence score
# =============================================================================================
with tab2:
    st.markdown(
        "Squash all the token probabilities of an answer into **one number** you can act on, and flag the "
        "individual tokens that stand out as unsure. This is notebook 1's `calculate_response_confidence` "
        "and `analyze_token_confidence`. Run several questions to compare a well-known fact with a made-up one."
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Questions to compare", list(QUESTIONS), default=list(QUESTIONS)[:3], key="t2_q")
        custom = st.text_input("Add your own question (optional)", "", key="t2_custom")
        questions = [QUESTIONS[q] for q in chosen] + ([custom.strip()] if custom.strip() else [])
    with c2:
        weight = st.slider("Weight on the average (w)", 0.0, 1.0, DEFAULT_WEIGHT, 0.05, key="t2_w",
                           help="score = w × average + (1 − w) × minimum. Notebook: w = 0.7.")
        low_fraction = st.slider("Low-confidence threshold (fraction of the average)", 0.0, 1.0,
                                 DEFAULT_LOW_FRACTION, 0.05, key="t2_f",
                                 help="A token is flagged if its probability < this × the answer's average. "
                                      "Notebook: 0.5.")
        top_k2 = st.slider("Top-k alternatives per step", 1, 10, 5, key="t2_k")
        max_new2 = st.slider("Max answer length (tokens)", 20, 300, 150, key="t2_len")
    st.caption("Moving the first two sliders re-scores instantly; only *Run* calls the model.")

    if st.button("Run", type="primary", key="t2_go", disabled=not questions):
        cache = st.session_state.setdefault("t2_runs", {})
        lm = get_model(model_id)
        bar = st.progress(0.0)
        for n, q in enumerate(questions):
            key = (model_id, q, max_new2, top_k2)
            if key not in cache:
                bar.progress(n / len(questions), text=f"Answering: {q[:70]}…")
                cache[key] = answer_question(lm, q, max_new2, top_k2)
        bar.empty()
        st.session_state["t2_shown"] = [(model_id, q, max_new2, top_k2) for q in questions]

    runs = [st.session_state.get("t2_runs", {}).get(k) for k in st.session_state.get("t2_shown", [])]
    runs = [r for r in runs if r and r.steps]
    if runs:
        scored = [(r, response_confidence(r.steps, weight, low_fraction)) for r in runs]
        st.subheader("Side by side")
        st.dataframe(pd.DataFrame([{
            "question": r.question,
            "answer": r.answer,
            "tokens": len(r.steps),
            "average": rc.average,
            "minimum": rc.minimum,
            "weakest token": f"{rc.weakest.label} (step {rc.weakest.index})",
            "response confidence score": rc.score,
            "low-confidence tokens": len(rc.low_tokens),
        } for r, rc in scored]), column_config={
            "average": PROB_COLUMN, "minimum": PROB_COLUMN, "response confidence score": PROB_COLUMN,
        }, hide_index=True, width="stretch")

        st.subheader("Each answer in detail")
        legend()
        for r, rc in scored:
            with st.expander(f"{rc.score:.2%}: {r.question}", expanded=len(scored) == 1):
                st.markdown(heatmap_html(r.steps, rc.low_threshold), unsafe_allow_html=True)
                st.caption("Red outline = low-confidence token.")
                st.markdown(
                    f"**How the score is computed**\n\n"
                    f"- average of {len(r.steps)} token probabilities = **{rc.average:.4f}**\n"
                    f"- minimum = **{rc.minimum:.4f}** (token `{rc.weakest.label}` at step {rc.weakest.index})\n"
                    f"- score = {rc.weight:.2f} × {rc.average:.4f} + {1 - rc.weight:.2f} × {rc.minimum:.4f} "
                    f"= **{rc.score:.4f}**\n"
                    f"- low-confidence threshold = {low_fraction:.2f} × {rc.average:.4f} = **{rc.low_threshold:.4f}**"
                )
                if rc.low_tokens:
                    st.markdown(f"**{len(rc.low_tokens)} low-confidence tokens**")
                    st.dataframe(step_table(rc.low_tokens), column_config={"token probability": PROB_COLUMN},
                                 hide_index=True, width="stretch")
                else:
                    st.markdown("No token falls below the threshold.")
        st.info(
            "**Remember:** a low score means the model was *unsure*, not that it was *wrong*. Many low tokens "
            "are just wording choices (\"compositions\" vs \"works\"); others sit right on a made-up fact "
            "(a name, a year). Read the flagged tokens, not just the score. Section 3 of the guide walks "
            "through both kinds."
        )


# =============================================================================================
# ③ Self-check on receipts
# =============================================================================================
def rows_to_frame(rows: list[ReceiptRow]) -> pd.DataFrame:
    return pd.DataFrame([r.values() for r in rows], columns=list(FIELDS), dtype="float64")


def frame_to_rows(df: pd.DataFrame) -> list[ReceiptRow]:
    rows = []
    for _, r in df.iterrows():
        vals = [None if pd.isna(r[f]) else float(r[f]) for f in FIELDS]
        if any(v is not None for v in vals):
            rows.append(ReceiptRow(*vals))
    return rows


with tab3:
    st.markdown(
        "The model fills in **missing receipt fields**. Each output line gets a **line confidence**: the "
        "lowest token probability in that line. Because we know the right answer, we can check whether "
        "low line confidence really points at wrong rows, then use it to decide which rows to "
        "**trust** and which to **send for review**. This is notebook 2."
    )
    with st.expander("The rules the model is given (notebook 2's system prompt)"):
        st.markdown(
            "- If **tax** is missing: tax = 9.21% × billed_amount\n"
            "- If **tip** is missing: tip = paid_amount − billed_amount − tax\n"
            "- If **paid_amount** is missing: paid_amount = billed_amount + tax + tip\n"
            "- No headers or explanations (the app adds one example line, because small models ignore this)."
        )

    c1, c2 = st.columns([3, 2])
    with c2:
        source = st.radio("Start from", ["Notebook's 6 rows", "Random rows"], key="t3_src")
        if source == "Random rows":
            n_rows = st.slider("How many rows", 4, 30, 12, key="t3_n",
                               help="The 0.5B model handles about a dozen rows per call; with more it often loses the format.")
            seed = st.number_input("Seed (same seed → same rows)", 0, 10_000, 1, key="t3_seed")
            rows = random_rows(n_rows, int(seed))
        else:
            rows = parse_input(NOTEBOOK_ROWS)
        receipt_model = st.selectbox("Model for this tab", MODELS, index=0, key="t3_model",
                                     help="1.5B does arithmetic better but needs ~3 GB of RAM and is ~3× slower.")
    with c1:
        st.markdown("**Input rows** (blank cell = missing; you can edit any cell or add rows)")
        edited = st.data_editor(rows_to_frame(rows), num_rows="dynamic", width="stretch",
                                key=f"t3_editor_{source}_{st.session_state.get('t3_n')}_{st.session_state.get('t3_seed')}")
    if st.button("Fill in the missing values", type="primary", key="t3_go"):
        with st.spinner("The model is filling in the rows (one call for all rows, as in the notebook)…"):
            st.session_state["t3"] = impute_receipts(get_model(receipt_model), frame_to_rows(edited))

    run = st.session_state.get("t3")
    if run:
        st.subheader("1 · What went in and what came out")
        a, b = st.columns(2)
        a.markdown("**Sent to the model**")
        a.code(run.input_csv, language="text")
        b.markdown(f"**{run.model_id} replied**")
        b.code(run.output or "(nothing)", language="text")
        if run.skipped:
            st.caption(f"Ignored {len(run.skipped)} output line(s) with no digits (blank lines, ``` fences or "
                       f"chatter) so they don't shift the rows: "
                       + ", ".join(f"`{l.text.strip() or '(blank)'}`" for l in run.skipped))

        st.subheader("2 · Line confidence of every row")
        st.markdown("Each line's **weakest token** (the one that sets its line confidence) is underlined in blue.")
        legend()
        weakest = {r.line.weakest.index for r in run.results if r.line}
        st.markdown(heatmap_html(run.steps, marked=weakest), unsafe_allow_html=True)

        st.subheader("3 · Was the model right?")
        c1, c2 = st.columns(2)
        threshold = c1.slider("Review threshold: send a row for review if its line confidence is below",
                              0.0, 1.0, 0.5, 0.01, key="t3_th")
        tolerance = c2.number_input("A row counts as wrong if any field is off by more than ($)",
                                    0.0, 100.0, 0.01, 0.01, key="t3_tol")
        table = []
        for r in run.results:
            wrong = r.is_wrong(tolerance)
            weak = r.line.weakest if r.line else None
            table.append({
                "row": r.row_no,
                "input": r.given.as_csv() if r.given else "—",
                "model output": r.line.text.strip() if r.line else "—",
                "correct": r.correct.as_csv() if r.correct else "(can't tell)",
                "line confidence": r.line_confidence,
                "weakest token": f"'{weak.label}' (alts: {alternatives_text(weak, 3)})" if weak else "—",
                "gap from correct ($)": r.truth_gap,
                "adds up? error ($)": r.verification_error,
                "notebook's error ($)": r.notebook_verification_error,
                "changed a given value": ", ".join(r.changed_given),
                "status": r.status,
                "truth": "—" if wrong is None else ("❌ wrong" if wrong else "✔️ right"),
                "decision": ("—" if r.given is None else
                             "🔍 review" if (not r.parsed or r.line_confidence < threshold) else "✅ accept"),
            })
        st.dataframe(pd.DataFrame(table), column_config={"line confidence": PROB_COLUMN},
                     hide_index=True, width="stretch")
        with st.expander("What do the three error columns mean?"):
            st.markdown(
                "- **gap from correct**: the largest difference between the model's row and the true row "
                "(computed exactly from the rules above). This is what we really care about.\n"
                "- **adds up? error**: |billed + tax + tip − paid| using the model's *own* numbers. A row can "
                "add up perfectly and still be wrong (e.g. a copied row).\n"
                "- **notebook's error**: the notebook's `verify_row`, |round(billed × 1.0921, 2) + tip − paid|. "
                "It ignores the tax the model wrote and assumes 9.21% tax, so it flags rows whose given tax "
                "differs from 9.21%, and misses a wrong tax if tip and paid compensate."
            )

        st.subheader("4 · Does low line confidence point at wrong rows?")
        x_choice = st.radio("Horizontal axis", ["gap from correct ($)", "notebook's error ($)"], horizontal=True,
                            key="t3_x", help="The notebook plots its own verification error.")
        plot = pd.DataFrame([{
            "row": t["row"], "line confidence": t["line confidence"], "x": t[x_choice],
            "truth": t["truth"], "model output": t["model output"],
        } for t in table if t["line confidence"] is not None and t[x_choice] is not None])
        if not plot.empty:
            points = alt.Chart(plot).mark_circle(size=140).encode(
                x=alt.X("x:Q", title=x_choice, scale=alt.Scale(type="symlog")),
                y=alt.Y("line confidence:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("truth:N", scale=alt.Scale(domain=["✔️ right", "❌ wrong"],
                                                           range=["#2e7d32", "#c62828"])),
                tooltip=["row", "model output", "line confidence", "x"],
            )
            rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[6, 4], color="#1f4fd1").encode(y="y:Q")
            st.altair_chart((points + rule).properties(height=360), width="stretch")
            st.caption("Blue dashed line = review threshold: rows below it go to review. Horizontal axis is "
                       "symlog so a $0.09 error and a $300 error both fit. Unparseable rows have no point.")

        st.subheader("5 · How well does the threshold sort the rows?")
        t = triage(run.results, threshold, tolerance)
        fmt = lambda ids: f"**{len(ids)}**" + (f"  (rows {', '.join(map(str, ids))})" if ids else "")
        st.markdown(
            f"| | 🔍 sent to review | ✅ auto-accepted |\n|---|---|---|\n"
            f"| **❌ actually wrong** | caught: {fmt(t.caught)} | missed: {fmt(t.missed)} |\n"
            f"| **✔️ actually right** | false alarm: {fmt(t.false_alarms)} | correctly trusted: {fmt(t.accepted)} |"
        )
        wrong_total = len(t.caught) + len(t.missed)
        reviewed = len(t.caught) + len(t.false_alarms)
        if wrong_total:
            st.markdown(f"Caught **{len(t.caught)} of {wrong_total}** wrong rows while reviewing "
                        f"**{reviewed} of {reviewed + len(t.missed) + len(t.accepted)}** rows.")
        if t.unknown:
            st.caption(f"Rows {t.unknown}: the correct answer can't be worked out from the input, so they're "
                       "left out.")
        st.caption("Raise the threshold to catch more wrong rows at the cost of more false alarms; lower it "
                   "to review less but miss more. Unparseable or missing rows always go to review.")

        st.subheader("6 · Inspect one row's tokens")
        with_lines = [r for r in run.results if r.line]
        if with_lines:
            pick = st.selectbox("Row", [r.row_no for r in with_lines], key="t3_row",
                                format_func=lambda n: f"row {n}: {run.results[n].line.text.strip()}")
            line = run.results[pick].line
            st.markdown(f"Line confidence = min of these = **{line.line_confidence:.4f}** "
                        f"(token `{line.weakest.label}`). The ⏎ token counts toward the line it ends, as in "
                        "the notebook.")
            st.dataframe(step_table(line.tokens), column_config={"token probability": PROB_COLUMN},
                         hide_index=True, width="stretch")

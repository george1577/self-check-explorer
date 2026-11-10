"""Build the notebook's token tree (analyze_logprobs) as DOT text, which Streamlit renders in the browser."""

from .generation import TokenStep


def probability_color(p: float) -> str:
    """Red (0) → amber (0.5) → green (1), as a hex colour."""
    p = min(max(p, 0.0), 1.0)
    if p < 0.5:
        r, g = 230, int(80 + 2 * p * 130)
    else:
        r, g = int(230 - (p - 0.5) * 2 * 150), 210
    return f"#{r:02x}{g:02x}80"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def token_tree_dot(steps: list[TokenStep], max_tokens: int = 30, low_threshold: float | None = None) -> str:
    """Root → chosen token 1 → chosen token 2 → …, each chosen token's alternatives branching from the
    same parent as it (exactly like the notebook's graph). Chosen tokens are filled by probability;
    alternatives are grey. Tokens below low_threshold get a thick red border."""
    lines = [
        "digraph logprobs {",
        '  rankdir="LR";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  root [label="start", fillcolor="#dddddd"];',
    ]
    parent = "root"
    for s in steps[:max_tokens]:
        node = f"t{s.index}"
        border = ', penwidth=3, color="#c00000"' if low_threshold is not None and s.probability < low_threshold else ""
        label = _escape(f"P('{s.label}') = {100 * s.probability:.2f}%")
        lines.append(f'  {node} [label="{label}", fillcolor="{probability_color(s.probability)}"{border}];')
        lines.append(f"  {parent} -> {node} [penwidth=2];")
        for j, alt in enumerate(s.alternatives):
            alt_node = f"a{s.index}_{j}"
            alt_label = _escape(f"P('{alt.label}') = {100 * alt.probability:.2f}%")
            lines.append(f'  {alt_node} [label="{alt_label}", fillcolor="#f2f2f2", fontcolor="#666666"];')
            lines.append(f'  {parent} -> {alt_node} [style=dashed, color="#999999"];')
        parent = node
    if len(steps) > max_tokens:
        lines.append(f'  more [label="… {len(steps) - max_tokens} more tokens", shape=plaintext, style=""];')
        lines.append(f"  {parent} -> more [style=dotted];")
    lines.append("}")
    return "\n".join(lines)

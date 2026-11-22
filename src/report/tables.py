"""Write a table as CSV and as a LaTeX booktabs snippet you can \\input{} in Overleaf.

The .tex file is a complete `table` environment (caption + label), so in the report:
    \\usepackage{booktabs}          % in the preamble, once
    \\input{tables/table_metrics}   % where the table should go
"""
import csv
from pathlib import Path

_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                  "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


# Unicode that pdfLaTeX (Overleaf's default) can't typeset directly -> LaTeX commands.
_LATEX_UNICODE = {"×": r"\ensuremath{\times}", "→": r"\ensuremath{\rightarrow}", "∅": r"\ensuremath{\emptyset}",
                  "÷": r"\ensuremath{\div}", "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}",
                  "µ": r"\ensuremath{\mu}", "±": r"\ensuremath{\pm}", "–": "--", "—": "---",
                  "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''"}


def latex_escape(s) -> str:
    """Escape LaTeX specials (& % $ # _ { } ~ ^ \\) and convert non-ASCII symbols, so any text
    (including IAM labels such as '&' or '#') compiles with pdfLaTeX."""
    out = []
    for ch in str(s):
        if ch in _LATEX_SPECIAL:
            out.append(_LATEX_SPECIAL[ch])
        elif ch in _LATEX_UNICODE:
            out.append(_LATEX_UNICODE[ch])
        elif ch == '"':
            out.append(r"{\ttfamily\char34}")
        elif ch in "<>|":
            out.append(r"\ensuremath{" + ch + "}")
        else:
            out.append(ch)
    return "".join(out)


def fmt(v, digits=4):
    """Numbers -> short strings: ints with thousands separators, floats rounded."""
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_latex(rows, columns, caption, label, col_format=None, raw_cols=(), note=None, digits=4) -> str:
    """booktabs table. `raw_cols` are columns whose cells are already LaTeX (not escaped)."""
    col_format = col_format or ("l" + "r" * (len(columns) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{col_format}}}", r"\toprule",
             " & ".join(rf"\textbf{{{latex_escape(c)}}}" for c in columns) + r" \\", r"\midrule"]
    for r in rows:
        if r == "midrule":
            lines.append(r"\midrule"); continue
        cells = [fmt(r.get(c, ""), digits) for c in columns]
        cells = [cell if c in raw_cols else latex_escape(cell) for c, cell in zip(columns, cells)]
        if r.get("_bold"):                      # highlight a row (e.g. the chosen sweep setting)
            cells = [rf"\textbf{{{c}}}" if c else c for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip{{\footnotesize {latex_escape(note)}\par}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(rows, columns, out_dir, name, caption, label=None, **kw):
    """Save <name>.csv and <name>.tex. `rows` is a list of dicts (or "midrule"). Returns both paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = out_dir / f"{name}.csv", out_dir / f"{name}.tex"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in rows if r != "midrule")
    tex_path.write_text(to_latex(rows, columns, caption, label or f"tab:{name}", **kw))
    return [str(csv_path), str(tex_path)]


def csv_to_markdown(csv_path) -> str:
    """A saved CSV table as a Markdown table (to preview it in the notebook)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    cell = lambda c: c.replace("|", "\\|")
    head = "| " + " | ".join(map(cell, rows[0])) + " |\n|" + "---|" * len(rows[0])
    return head + "\n" + "\n".join("| " + " | ".join(map(cell, r)) + " |" for r in rows[1:])

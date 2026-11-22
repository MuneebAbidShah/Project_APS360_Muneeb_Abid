"""Character and word error rate (CER / WER).

Both are corpus-level: add up the edits over all samples, then divide by the
total length of the references. This weights long words more than short ones,
which is the standard way to report CER/WER (averaging per-word rates is not).
"""
try:
    import editdistance as _ed  # fast C implementation, if installed
except ImportError:  # pragma: no cover
    _ed = None


def edit_distance(a, b) -> int:
    """Levenshtein distance: fewest insertions, deletions and substitutions
    turning sequence `a` into `b`. Works on strings or lists of words."""
    if _ed is not None:
        return _ed.eval(a, b)
    return _edit_distance_py(a, b)


def _edit_distance_py(a, b) -> int:
    """Pure-Python dynamic programming version (one row at a time)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,             # delete x
                cur[j - 1] + 1,          # insert y
                prev[j - 1] + (x != y),  # substitute (free if equal)
            ))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """Total character edits / total reference characters."""
    assert len(refs) == len(hyps), "need one prediction per reference"
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    """Total word edits / total reference words (words split on whitespace).

    For our single-word images this is just the fraction of words that are
    not exactly right.
    """
    assert len(refs) == len(hyps), "need one prediction per reference"
    edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return edits / max(total, 1)

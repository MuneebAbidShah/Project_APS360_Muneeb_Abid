"""Error analysis: which characters get confused, and how CER depends on word length."""
from collections import Counter

from src.utils.metrics import cer, wer

EMPTY = "\u2205"   # the symbol used for "nothing" in insertions / deletions


def align(ref: str, hyp: str):
    """Levenshtein alignment of two strings. Returns a list of (op, ref_char, hyp_char)
    with op in {"=", "sub", "del", "ins"}; del = character missed, ins = extra character."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:   # walk back from the corner
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], EMPTY)); i -= 1
        else:
            ops.append(("ins", EMPTY, hyp[j - 1])); j -= 1
    return ops[::-1]


def char_confusions(refs, hyps):
    """Counter of (true char, predicted char) over all errors; EMPTY marks a missed/extra char."""
    c = Counter()
    for r, h in zip(refs, hyps):
        if r != h:
            c.update((a, b) for op, a, b in align(r, h) if op != "=")
    return c


def error_breakdown(refs, hyps):
    """Total substitutions / deletions / insertions (the three parts of CER)."""
    c = Counter()
    for r, h in zip(refs, hyps):
        if r != h:
            c.update(op for op, _, _ in align(r, h) if op != "=")
    return {"substitutions": c["sub"], "deletions": c["del"], "insertions": c["ins"],
            "reference_chars": sum(len(r) for r in refs)}


def word_mistakes(refs, hyps, k=15):
    """Most frequent (truth, prediction) pairs among wrong words."""
    return Counter((r, h) for r, h in zip(refs, hyps) if r != h).most_common(k)


LENGTH_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 9), (10, 99)]


def bin_name(lo, hi):
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 99 else f"{lo}-{hi}")


def cer_by_length(refs, hyps, bins=LENGTH_BINS):
    """CER and WER per word-length bin. Returns rows with bin, n, cer, wer."""
    rows = []
    for lo, hi in bins:
        pairs = [(r, h) for r, h in zip(refs, hyps) if lo <= len(r) <= hi]
        if pairs:
            rs, hs = zip(*pairs)
            rows.append({"length": bin_name(lo, hi), "n": len(pairs), "cer": cer(rs, hs), "wer": wer(rs, hs)})
    return rows

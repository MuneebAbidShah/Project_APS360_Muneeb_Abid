"""Report figures. Every function returns the saved [png, pdf] paths (see style.save_figure)."""
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.report.style import (FULL_WIDTH, HALF_WIDTH, GREEN, RED, BLUE, ORANGE, GREY, save_figure)


def read_history(history_csv):
    """history.csv written by fit() (one row per epoch, survives resumes)."""
    rows = list(csv.DictReader(open(history_csv)))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def _best_epoch(hist):
    return int(min(hist, key=lambda r: r["val_cer"])["epoch"])


def _loss_axes(ax, hist):
    ep = [r["epoch"] for r in hist]
    ax.plot(ep, [r["train_loss"] for r in hist], color=BLUE, label="train")
    ax.plot(ep, [r["val_loss"] for r in hist], color=ORANGE, label="validation")
    ax.axvline(_best_epoch(hist), color=GREY, ls=":", lw=1, label="best epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("CTC loss"); ax.set_title("Training and validation loss")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="best")


def _error_axes(ax, hist):
    ep = [r["epoch"] for r in hist]
    ax.plot(ep, [r["val_cer"] for r in hist], color=BLUE, marker="o", ms=2.5, label="val CER")
    ax.plot(ep, [r["val_wer"] for r in hist], color=ORANGE, marker="s", ms=2.5, label="val WER")
    b = _best_epoch(hist)
    best = next(r for r in hist if int(r["epoch"]) == b)
    ax.axvline(b, color=GREY, ls=":", lw=1, label=f"best: epoch {b}, CER {best['val_cer']:.3f}")
    ax.set_ylim(0, min(1.05, max(max(r["val_wer"] for r in hist) * 1.1, 0.1)))
    ax.set_xlabel("epoch"); ax.set_ylabel("error rate"); ax.set_title("Validation CER and WER")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="best")


def _int_epochs(*axes):
    from matplotlib.ticker import MaxNLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_training_curves(history_csv, out_dir):
    """Three files: combined two-panel figure, loss only, CER/WER only."""
    hist = read_history(history_csv)
    paths = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.1))
    _loss_axes(a1, hist); _error_axes(a2, hist); _int_epochs(a1, a2); fig.tight_layout()
    paths += save_figure(fig, out_dir, "fig_training_curves")
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.0)); _loss_axes(ax, hist); _int_epochs(ax); fig.tight_layout()
    paths += save_figure(fig, out_dir, "fig_loss_curve")
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.0)); _error_axes(ax, hist); _int_epochs(ax); fig.tight_layout()
    paths += save_figure(fig, out_dir, "fig_val_cer_wer_curve")
    return paths


def plot_prediction_grid(images, refs, hyps, out_dir, name, title=None, cols=3):
    """Word images (original resolution, uint8 arrays) with true and predicted text.
    Prediction is green when exactly right and red when wrong."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 0.95 * rows + (0.25 if title else 0)))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, r, h in zip(np.atleast_1d(axes).ravel(), images, refs, hyps):
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#cccccc"); s.set_linewidth(0.5)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_title(f"true: {r}", fontsize=7.5, loc="left", pad=2, parse_math=False)
        ax.set_xlabel(f"pred: {h if h else '(empty)'}", fontsize=7.5, loc="left", labelpad=2,
                      color=GREEN if r == h else RED, fontweight="bold", parse_math=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    return save_figure(fig, out_dir, name)


def plot_char_confusions(confusions: Counter, out_dir, name="fig_char_confusions", k=15, title=None):
    """Horizontal bars: the k most frequent character errors (true -> predicted)."""
    top = confusions.most_common(k)[::-1]
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 0.17 * max(len(top), 1) + 0.6))
    labels = [f"{a!s} \u2192 {b!s}" for (a, b), _ in top]
    ax.barh(range(len(top)), [n for _, n in top], color=BLUE)
    ax.set_yticks(range(len(top)), labels, fontfamily="DejaVu Sans Mono")
    for t in ax.get_yticklabels():
        t.set_parse_math(False) if hasattr(t, "set_parse_math") else None
    ax.set_xlabel("count"); ax.grid(axis="y", visible=False)
    ax.set_title(title or "Most common character errors", loc="left")
    fig.tight_layout()
    return save_figure(fig, out_dir, name)


def plot_label_length_hist(texts, out_dir, name="fig_label_length_hist", split="train"):
    lengths = [len(t) for t in texts]
    c = Counter(lengths)
    xs = list(range(1, max(lengths) + 1))
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.9))
    ax.bar(xs, [c[x] for x in xs], color=BLUE, width=0.85)
    m = float(np.mean(lengths))
    ax.axvline(m, color=RED, ls="--", lw=1, label=f"mean {m:.1f}")
    ax.set_xlabel("word length (characters)"); ax.set_ylabel("words")
    ax.set_title(f"Label lengths ({split}, n={len(lengths):,})", loc="left")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_figure(fig, out_dir, name)


def plot_cer_by_length(systems: dict, out_dir, name="fig_cer_by_length", title=None):
    """Grouped bars of CER per word-length bin. `systems` maps a name to rows from
    errors.cer_by_length (all computed on the same words)."""
    names = list(systems)
    bins = [r["length"] for r in systems[names[0]]]
    ns = {r["length"]: r["n"] for r in systems[names[0]]}
    x = np.arange(len(bins)); w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH if len(names) > 1 else HALF_WIDTH * 1.25, 2.0))
    colors = [BLUE, ORANGE, GREEN, RED]
    for i, s in enumerate(names):
        vals = {r["length"]: r["cer"] for r in systems[s]}
        ax.bar(x + (i - (len(names) - 1) / 2) * w, [vals.get(b, 0) for b in bins], w,
               label=s, color=colors[i % len(colors)])
    ax.set_xticks(x, [f"{b}\n({ns[b]:,})" for b in bins], fontsize=6.5)
    ax.set_xlabel("word length in characters (number of test words)")
    ax.set_ylabel("CER"); ax.grid(axis="x", visible=False)
    ax.set_title(title or "CER by word length (test)", loc="left")
    if len(names) > 1:
        top = max(r["cer"] for rows in systems.values() for r in rows)
        ax.set_ylim(0, top * 1.28)       # headroom so the legend sits above the bars
        ax.legend(frameon=False, ncol=len(names), loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, name)


# ------------------------------------------------------------------ block diagrams

def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                fc=color, ec="#333333", lw=0.6))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=6.6, fontweight="bold")
    ax.text(x + w / 2, y + (h - 0.1) / 2, sub, ha="center", va="center", fontsize=5.9, color="#222222",
            linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333",
                                                         shrinkA=0, shrinkB=0, mutation_scale=7))


def _row(ax, boxes, y, h, x0=0.005, x1=0.995, gap=0.022):
    w = (x1 - x0 - gap * (len(boxes) - 1)) / len(boxes)
    xs = [x0 + i * (w + gap) for i in range(len(boxes))]
    for x, (t, s, c) in zip(xs, boxes):
        _box(ax, x, y, w, h, t, s, c)
    for i in range(len(boxes) - 1):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    return xs, w


def crnn_shapes(model_cfg, num_classes, height=32):
    """Tensor shape after each stage, as text, computed from the model settings."""
    c = list(model_cfg.cnn_channels)
    pools = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]
    h, wdiv, out = height, 1, []
    for ch, (ph, pw) in zip(c, pools):
        h //= ph; wdiv *= pw
        out.append((ch, h, wdiv))
    return out


def draw_crnn_diagram(model_cfg, num_classes, out_dir, name="fig_crnn_architecture", height=32):
    """Block diagram of the CRNN with the tensor shape after every stage (W = image width)."""
    shapes = crnn_shapes(model_cfg, num_classes, height)
    pools = ["2×2", "2×2", "2×1", "2×1", "2×1"]
    wtxt = lambda d: "W" if d == 1 else f"W/{d}"
    conv = "#cfe2f3"; seq = "#d9ead3"; out = "#fce5cd"; inp = "#eeeeee"
    row1 = [("Input", f"grayscale\nimage\n1×{height}×W", inp)] + [
        (f"Conv {i + 1}", f"{ch} filters\npool {p}\n→ {ch}×{h}×{wtxt(d)}", conv)
        for i, ((ch, h, d), p) in enumerate(zip(shapes, pools))]
    T = wtxt(shapes[-1][2])
    H2 = 2 * model_cfg.lstm_hidden
    row2 = [("Reshape", f"to a sequence\nT = {T} steps\n→ T×{shapes[-1][0]}", seq),
            ("BiLSTM", f"{model_cfg.lstm_layers} layers, {model_cfg.lstm_hidden}\nper direction\n→ T×{H2}", seq),
            ("Linear", f"{H2} → {num_classes}\nlog-softmax\n→ T×{num_classes}", out),
            ("CTC", "loss in training\ngreedy decoding\nat test time", out),
            ("Output text", "e.g.\n\"which\"", inp)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    h = 0.38
    xs1, w1 = _row(ax, row1, 0.53, h)
    xs2, w2 = _row(ax, row2, 0.02, h, x0=0.06, x1=0.94)
    xa, xb, ym = xs1[-1] + w1 / 2, xs2[0] + w2 / 2, 0.465   # elbow arrow: last conv block -> reshape
    ax.plot([xa, xa, xb], [0.53, ym, ym], color="#333333", lw=0.7)
    _arrow(ax, xb, ym, xb, 0.02 + h)
    ax.text(0.005, 0.995, f"Each conv block: 3×3 convolution, BatchNorm, ReLU, max-pool.  "
                          f"CNN: height {height} → 1, width ÷ {shapes[-1][2]}", fontsize=6.3, va="top", color=GREY)
    return save_figure(fig, out_dir, name)


def draw_tesseract_pipeline(out_dir, name="fig_tesseract_pipeline", border=16, psm=8, oem=1):
    """Block diagram of the Tesseract baseline."""
    g = "#eeeeee"; b = "#cfe2f3"; o = "#fce5cd"
    boxes = [("Word image", "IAM crop at\noriginal resolution\n(grayscale)", g),
             ("Preprocess", f"pad a {border}-px\nwhite border\non every side", b),
             ("Tesseract", f"LSTM engine\n--oem {oem} --psm {psm}\n(single word, eng)", b),
             ("Normalize", "strip whitespace\nand newlines,\nASCII quotes", o),
             ("Score", "CER / WER vs\nIAM label (same\ncode as CRNN)", o)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _row(ax, boxes, 0.03, 0.94, gap=0.04)
    return save_figure(fig, out_dir, name)


def plot_preprocessing_examples(originals, processed, texts, out_dir, name="fig_preprocessing_examples", height=32):
    """Cleaned training samples: the original IAM crop (left) and the CRNN input (right):
    grayscale, resized to `height` px keeping the aspect ratio, normalized to [-1, 1]."""
    n = len(originals)
    fig, axes = plt.subplots(n, 2, figsize=(FULL_WIDTH, 0.62 * n + 0.3),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    axes = np.atleast_2d(axes)
    for (a, b), o, p, t in zip(axes, originals, processed, texts):
        a.imshow(o, cmap="gray", vmin=0, vmax=255)
        a.set_ylabel(f"\"{t}\"", rotation=0, ha="right", va="center", fontsize=8, parse_math=False)
        b.imshow(p, cmap="gray", vmin=-1, vmax=1)
        a.set_xlabel(f"{o.shape[1]}×{o.shape[0]} px", fontsize=6.5, labelpad=1, color=GREY)
        b.set_xlabel(f"{p.shape[1]}×{p.shape[0]} px", fontsize=6.5, labelpad=1, color=GREY)
        for ax in (a, b):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#cccccc"); s.set_linewidth(0.5)
    axes[0, 0].set_title("Original IAM word crop", fontsize=8)
    axes[0, 1].set_title(f"CRNN input: {height} px high, aspect kept, values in [-1, 1]", fontsize=8)
    fig.tight_layout(h_pad=0.3, w_pad=1.0)
    return save_figure(fig, out_dir, name)


def plot_sweep(rows, out_dir, name="fig_sweep", chosen=None, title=None):
    """Best validation CER of every sweep run (best at the top); the chosen run is highlighted."""
    rows = sorted(rows, key=lambda r: (r["best_val_cer"], r["best_val_wer"], r["name"]))[::-1]
    labels = [f"lr {r['lr']:g}, h {r['lstm_hidden']}, L {r['lstm_layers']}, d {r['dropout']:g}, b {r['batch_size']}"
              for r in rows]
    vals = [r["best_val_cer"] for r in rows]
    colors = [ORANGE if r["name"] == chosen else BLUE for r in rows]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.19 * len(rows) + 0.7))
    ax.barh(range(len(rows)), vals, color=colors)
    ax.set_yticks(range(len(rows)), labels, fontsize=6.5, fontfamily="DejaVu Sans Mono")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
    ax.set_xlabel("best validation CER (lower is better)"); ax.grid(axis="y", visible=False)
    ep = rows[0]["epochs"] if rows else 0
    fr = rows[0]["train_frac"] if rows else 0
    ax.set_title(title or f"Hyperparameter sweep ({len(rows)} runs, {ep} epochs on {fr:.0%} of train; "
                          "orange = chosen)", loc="left")
    fig.tight_layout()
    return save_figure(fig, out_dir, name)


def draw_emnist_pipeline(out_dir, name="fig_emnist_pipeline"):
    """Block diagram of the EMNIST character-classifier baseline."""
    g = "#eeeeee"; b = "#cfe2f3"; o = "#fce5cd"
    boxes = [("Word", "IAM crop,\noriginal size", g),
             ("Binarize", "Otsu, then\ndeslant and\ncomponents", b),
             ("Split", "cut wide\nblobs at ink\nminima", b),
             ("Normalize", "28x28,\ncentered,\nEMNIST-like", b),
             ("Char CNN", "EMNIST only,\n62 classes", o),
             ("Join", "punctuation\nrules; same\nCER / WER", o)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _row(ax, boxes, 0.03, 0.94, gap=0.03)
    return save_figure(fig, out_dir, name)


def plot_segmentation_examples(examples, out_dir, name="fig_emnist_segmentation"):
    """Each row: the word with its character boxes (red = letter crop, blue = punctuation rule),
    then every normalized 28x28 crop with the classifier's prediction under it.
    `examples`: list of dicts with gray, pieces, crops, preds (one char per piece), ref, hyp."""
    from matplotlib.patches import Rectangle
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(FULL_WIDTH, 0.85 * n + 0.25), gridspec_kw={"width_ratios": [1, 1.5]})
    axes = np.atleast_2d(axes)
    for (a, b), ex in zip(axes, examples):
        a.imshow(ex["gray"], cmap="gray", vmin=0, vmax=255)
        for q in ex["pieces"]:
            a.add_patch(Rectangle((q["x0"] - 0.5, q["y0"] - 0.5), q["x1"] - q["x0"], q["y1"] - q["y0"], fill=False,
                                  lw=0.7, ec=RED if q["kind"] == "char" else BLUE))
        ok = ex["ref"] == ex["hyp"]
        a.set_title(f"true: {ex['ref']}   pred: {ex['hyp'] or '(empty)'}", fontsize=7, loc="left", pad=2,
                    color=GREEN if ok else RED, parse_math=False)
        crops = list(ex["crops"]) or [np.zeros((28, 28), np.uint8)]
        gap = np.full((28, 4), 90, np.uint8)
        strip = np.hstack([x for c in crops for x in (c, gap)][:-1])
        b.imshow(strip, cmap="gray", vmin=0, vmax=255)
        chars = [p for p, q in zip(ex["preds"], ex["pieces"]) if q["kind"] == "char"]
        b.set_xticks([16 + 32 * i - 2 for i in range(len(chars))], chars, fontsize=7)
        b.tick_params(length=0, pad=1)
        a.set_xticks([]); a.set_yticks([]); b.set_yticks([])
        for ax in (a, b):
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#cccccc"); s.set_linewidth(0.5)
    axes[0, 1].set_title("normalized 28x28 crops and predicted characters", fontsize=7, loc="left", pad=2)
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    return save_figure(fig, out_dir, name)

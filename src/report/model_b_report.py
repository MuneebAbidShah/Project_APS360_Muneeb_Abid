"""Step 6 (Model B = CRNN trained with camera-style augmentation): report figures, tables, numbers.

Everything goes to outputs/model_b/ (its own folder), never to Step 2's outputs/report/.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import augment as A
from src.report import figures, tables
from src.report.style import FULL_WIDTH, save_figure
from src.utils.metrics import cer, wer

pct = lambda x: f"{100 * x:.2f}"

# one distortion at a time, for the "what each augmentation does" figure
SINGLE = [("perspective", lambda im, r: A.perspective_warp(im, r, 0.12)),
          ("rotation", lambda im, r: A.rotate(im, r, 5.0)),
          ("motion blur", lambda im, r: A.motion_blur(im, r, 7)),
          ("uneven light", A.uneven_lighting), ("shadow", A.shadow), ("contrast", A.contrast),
          ("noise", lambda im, r: A.noise(im, r, 10.0)), ("JPEG", lambda im, r: A.jpeg(im, r, (25, 40)))]


def _show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def save_augmentation_examples(out_dir, aug_ds, n_words=4, n_aug=4, seed=0):
    """Two figures: (1) a few training words, clean + random augmented versions (exactly what
    Model B trains on, before the 32-px resize); (2) one word with each distortion on its own."""
    rng = np.random.default_rng(seed)
    idx = [int(i) for i in rng.choice(len(aug_ds), size=min(n_words, len(aug_ds)), replace=False)]
    fig, axes = plt.subplots(len(idx), n_aug + 1, figsize=(FULL_WIDTH, 0.55 * len(idx) + 0.3), squeeze=False)
    for r, i in enumerate(idx):
        _show(axes[r][0], aug_ds.image(i, augment=False), "clean (IAM)" if r == 0 else None)
        for k in range(n_aug):
            _show(axes[r][k + 1], aug_ds.image(i, epoch=k + 1, augment=True), f"augmented {k + 1}" if r == 0 else None)
    fig.tight_layout(pad=0.3)
    paths = save_figure(fig, out_dir, "fig_model_b_augmentation_examples")

    img = aug_ds.image(idx[0], augment=False)
    cols = 3
    rows = int(np.ceil((len(SINGLE) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(FULL_WIDTH, 0.75 * rows + 0.2), squeeze=False)
    flat = [a for row in axes for a in row]
    _show(flat[0], img, "clean")
    for ax, (name, fn) in zip(flat[1:], SINGLE):
        _show(ax, A._u8(fn(img, np.random.default_rng(seed + 1))), name)
    for ax in flat[len(SINGLE) + 1:]:
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    paths += save_figure(fig, out_dir, "fig_model_b_augmentation_types")
    return paths


def save_curves(out_dir, history_csv):
    """Step 2's curve figures, renamed fig_model_b_* so they can't be mixed up with Model A's."""
    out_dir = Path(out_dir)
    paths = figures.plot_training_curves(history_csv, out_dir)
    renamed = []
    for p in paths:
        q = Path(p).with_name(Path(p).name.replace("fig_", "fig_model_b_", 1))
        os.replace(p, q)
        renamed.append(str(q))
    return renamed


def save_metrics(out_dir, preds, eval_results, settings, run_dir, aug_cfg, model_a_metrics=None):
    """metrics.json + table_model_b_metrics (val/test on CLEAN IAM) + table_model_a_vs_b if Model A's
    numbers exist. Returns the numbers dict."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    nums = {"run_dir": str(run_dir), "settings": settings, "augmentation": aug_cfg,
            "note": "CER/WER on clean IAM val/test (no augmentation at evaluation); webcam set not scored here."}
    rows = []
    for split in ("val", "test"):
        p = preds[split]
        refs, hyps = [r["ref"] for r in p], [r["hyp"] for r in p]
        c, w = cer(refs, hyps), wer(refs, hyps)
        nums[split] = {"cer": c, "wer": w, "words": len(p), "loss": eval_results[split]["loss"],
                       "ms_per_word_batched": eval_results[split]["ms_per_word"]}
        rows.append({"Split": split, "Words": len(p), "CER (%)": pct(c), "WER (%)": pct(w)})
    (out_dir / "model_b_metrics.json").write_text(json.dumps(nums, indent=2, default=str))
    paths = tables.write_table(rows, list(rows[0]), out_dir, "table_model_b_metrics",
                               "Model B (CRNN trained with camera-style augmentation) on the clean IAM validation "
                               "and test splits (writer-independent, greedy decoding).", "tab:model_b_metrics")
    if model_a_metrics:
        comp = []
        for name, m in (("Model A (no augmentation)", model_a_metrics), ("Model B (augmentation)", nums)):
            comp.append({"Model": name,
                         "Val CER (%)": pct(m["val"]["cer"]), "Val WER (%)": pct(m["val"]["wer"]),
                         "Test CER (%)": pct(m["test"]["cer"]), "Test WER (%)": pct(m["test"]["wer"])})
        paths += tables.write_table(comp, list(comp[0]), out_dir, "table_model_a_vs_b",
                                    "Same CRNN and hyperparameters, trained without (A) and with (B) camera-style "
                                    "augmentation, both evaluated on clean IAM.", "tab:model_a_vs_b")
        nums["model_a"] = {s: model_a_metrics[s] for s in ("val", "test")}
    return nums, paths


def load_model_a_metrics(run_dir_a):
    """Model A's val/test numbers from Step 2's runs/<RUN_NAME>/metrics.json (read only), or None."""
    p = Path(run_dir_a) / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m if all(s in m and "cer" in m[s] for s in ("val", "test")) else None

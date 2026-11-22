"""Plots for the report: training curves and example predictions."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history_csv, out_png):
    rows = list(csv.DictReader(open(history_csv)))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    a1.plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("CTC loss"); a1.set_title("Loss"); a1.legend()
    a2.plot(ep, [float(r["val_cer"]) for r in rows], label="val CER")
    a2.plot(ep, [float(r["val_wer"]) for r in rows], label="val WER")
    a2.set_xlabel("epoch"); a2.set_ylabel("error rate"); a2.set_title("Validation error"); a2.legend()
    a2.set_ylim(0, max(1.0, max(float(r["val_wer"]) for r in rows)) if rows else 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def plot_predictions(images, refs, hyps, out_png, n=12):
    """Images with 'truth -> prediction' titles; green if right, red if wrong."""
    n = min(n, len(refs))
    fig, axes = plt.subplots(n, 1, figsize=(6, 0.85 * n + 0.4))
    axes = [axes] if n == 1 else axes
    for ax, img, r, h in zip(axes, images[:n], refs[:n], hyps[:n]):
        ax.imshow(img[0].numpy(), cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"{r!r} -> {h!r}", fontsize=8, loc="left", color="green" if r == h else "red")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png

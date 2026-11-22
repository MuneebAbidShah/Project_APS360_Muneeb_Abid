"""Evaluate a checkpoint: CER/WER on a split, per-word speed, and every prediction.

  python scripts/evaluate.py --checkpoint runs/crnn/best.pt --data-root /content/iam --split test

Writes <out-dir>/<split>_metrics.json and <split>_predictions.csv (ref, hyp, edits),
and prints the most common mistakes, which is useful for the report's error analysis.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.data import load_data
from src.training.engine import evaluate, get_device, load_model, make_loader
from src.utils.metrics import edit_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split-dir", default="default")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=None, help="default: the checkpoint's folder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-check-images", action="store_true")
    a = ap.parse_args()

    device = get_device()
    model, charset, ck = load_model(a.checkpoint, device)
    print(f"checkpoint epoch {ck['epoch']} (val CER {ck['best_cer']:.4f})")
    datasets, _, _ = load_data(a.data_root, (a.split,), preload=False,
                               check_images=not a.no_check_images, split_dir=a.split_dir)
    loader, _ = make_loader(datasets[a.split], a.batch_size, shuffle=False)
    res = evaluate(model, loader, charset, device, amp=device.type == "cuda")

    out = Path(a.out_dir or Path(a.checkpoint).parent); out.mkdir(parents=True, exist_ok=True)
    metrics = {k: res[k] for k in ("cer", "wer", "loss", "n", "ms_per_word")}
    metrics.update(split=a.split, checkpoint=str(a.checkpoint), epoch=ck["epoch"])
    (out / f"{a.split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / f"{a.split}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "prediction", "edits"])
        for r, h in zip(res["refs"], res["hyps"]):
            w.writerow([r, h, edit_distance(r, h)])

    print(f"\n{a.split}: {res['n']} words | CER {res['cer']:.4f} | WER {res['wer']:.4f} | "
          f"{res['ms_per_word']:.2f} ms/word (batched, {device})")
    wrong = [(r, h) for r, h in zip(res["refs"], res["hyps"]) if r != h]
    print(f"{len(wrong)} words wrong. Most common mistakes:")
    for (r, h), n in Counter(wrong).most_common(10):
        print(f"  {n:4d}x  {r!r} -> {h!r}")
    print(f"saved {out}/{a.split}_metrics.json and {a.split}_predictions.csv")


if __name__ == "__main__":
    main()

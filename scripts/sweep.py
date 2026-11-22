"""Hyperparameter sweep (same 12 settings as the Step 2 notebook), ranked by validation CER.

  python scripts/sweep.py --data-root /content/iam --out-dir runs/sweep --epochs 8 --train-frac 0.3

See src/training/sweep.py for the design. Results are appended to <out-dir>/sweep_results.csv as
each run finishes; re-running skips runs already there. The winner (lowest val CER) is printed
and is what the notebook's Train cell (choose_settings) uses. The test set is never touched.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.data import load_data
from src.training.sweep import choose_settings, run_sweep, sweep_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split-dir", default="default")
    ap.add_argument("--out-dir", default="outputs/runs/sweep")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--only", type=int, default=None, help="run only the first N settings (quick tests)")
    ap.add_argument("--hidden-scale", type=float, default=1.0, help=argparse.SUPPRESS)  # tests: smaller models
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--no-check-images", action="store_true")
    a = ap.parse_args()

    datasets, charset, _ = load_data(a.data_root, ("train", "val"), preload=not a.no_preload,
                                     check_images=not a.no_check_images, split_dir=a.split_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs()[: a.only] if a.only else sweep_configs()
    if a.hidden_scale != 1.0:
        configs = [{**c, "lstm_hidden": max(8, int(c["lstm_hidden"] * a.hidden_scale))} for c in configs]
    csv_path = out / "sweep_results.csv"
    ranked = run_sweep(datasets, charset, csv_path, out, configs, epochs=a.epochs, train_frac=a.train_frac,
                       max_batches=a.max_batches)
    print("\nRanked by val CER:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['name']:28s} CER {r['best_val_cer']:.4f}  WER {r['best_val_wer']:.4f}  ({r['minutes']:.1f} min)")
    settings, info = choose_settings(csv_path)
    print(f"\nTrain would use ({info['source']}): {settings}\n  {info['reason']}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

"""Train the CRNN on IAM words.

  python scripts/train.py --data-root /content/iam --out-dir runs/crnn
  python scripts/train.py --data-root data/fake_iam --epochs 2 --no-preload   # quick CPU check

Re-running with the same --out-dir resumes from last.pt (handy when Colab disconnects).
Pass --fresh to start over.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig, TrainConfig
from src.training.data import load_data
from src.training.engine import fit


def add_common_args(ap):
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split-dir", default="default", help="'none' = automatic writer-disjoint split")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--hidden", type=int, default=ModelConfig.lstm_hidden)
    ap.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    ap.add_argument("--patience", type=int, default=TrainConfig.patience)
    ap.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--no-check-images", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)


def configs_from_args(a, out_dir):
    mc = ModelConfig(lstm_hidden=a.hidden, dropout=a.dropout)
    # Preloaded images are already in RAM, so worker processes would only add overhead.
    workers = a.num_workers if a.no_preload else 0
    tc = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, patience=a.patience,
                     num_workers=workers,
                     train_frac=a.train_frac, max_batches=a.max_batches, amp=not a.no_amp,
                     preload=not a.no_preload, seed=a.seed, out_dir=Path(out_dir))
    return mc, tc


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out-dir", default="outputs/runs/crnn")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start again")
    a = ap.parse_args()
    datasets, charset, _ = load_data(a.data_root, ("train", "val"), preload=not a.no_preload,
                                     check_images=not a.no_check_images, split_dir=a.split_dir)
    mc, tc = configs_from_args(a, a.out_dir)
    result = fit(datasets, charset, mc, tc, resume=not a.fresh)
    print(f"\nbest val CER {result['best_val_cer']:.4f}  ->  {result['out_dir']}/best.pt")


if __name__ == "__main__":
    main()

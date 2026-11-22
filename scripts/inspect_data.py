"""Print dataset statistics and save a picture of one padded batch.

Usage:
  python scripts/inspect_data.py --data-root data/fake_iam
  python scripts/inspect_data.py --data-root /content/iam --out outputs/iam_batch.png
"""
import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config import DataConfig
from src.data.charset import Charset
from src.data.iam import SPLITS, collate_fn, make_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="IAM folder (default: config.DATA_ROOT)")
    ap.add_argument("--split-dir", default=None, help="folder with split lists; 'none' = auto split")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="outputs/sample_batch.png")
    ap.add_argument("--stats-out", default=None, help="also save the stats as JSON here")
    ap.add_argument("--no-check-images", action="store_true", help="skip opening every image (faster)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = DataConfig()
    if args.data_root:
        cfg.iam_root = Path(args.data_root)
    if args.split_dir:
        cfg.split_dir = None if args.split_dir.lower() == "none" else Path(args.split_dir)
    cfg.check_images = not args.no_check_images
    charset = Charset()

    datasets, stats = make_datasets(cfg, charset)

    print("\n=== Filtering (numbers for the report) ===")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")

    print("\n=== Splits ===")
    writer_sets, unknown = {}, 0
    has_writers = any(i.writer is not None for s in SPLITS for i in datasets[s].items)
    for s in SPLITS:
        items = datasets[s].items
        writer_sets[s] = {i.writer for i in items if i.writer is not None}
        unknown += sum(i.writer is None for i in items)
        lens = [len(i.text) for i in items] or [0]
        n_w = f"{len(writer_sets[s]):4d}" if has_writers else "   ?"
        print(f"  {s:5s}: {len(items):6d} words | {n_w} writers | "
              f"label len mean {statistics.mean(lens):.1f}, max {max(lens)}")

    # Writer-disjoint check: no writer may appear in two splits.
    overlap = None
    if not has_writers:
        print("  writer IDs unavailable; Aachen splits are writer-independent by design"
              if "split files" in str(stats.get("split_method")) else
              "  writer IDs unavailable; the split groups whole forms by prefix instead")
    else:
        overlap = (writer_sets["train"] & writer_sets["val"]) | (writer_sets["train"] & writer_sets["test"]) \
            | (writer_sets["val"] & writer_sets["test"])
        print(f"  writer IDs from: {stats.get('writer_source')}")
        print(f"  writers shared between splits: {len(overlap)} (should be 0)")
        if unknown:
            print(f"  ({unknown} words have no writer ID and are left out of this check)")

    if stats.get("malformed_line"):
        print(f"\n  NOTE: {stats['malformed_line']} line(s) of words.txt had no label and were skipped.")
        print("  If it's the last line, your words.txt was cut off: every word after it is missing.")

    if args.stats_out:
        summary = dict(stats, split_sizes={s: len(datasets[s]) for s in SPLITS},
                       split_writers={s: len(writer_sets[s]) for s in SPLITS},
                       writers_shared_between_splits=None if overlap is None else len(overlap))
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(summary, indent=2))
        print(f"  stats saved to {Path(args.stats_out).resolve()}")

    train_chars = Counter("".join(i.text for i in datasets["train"].items))
    print(f"\n  charset size: {len(charset.chars)} chars + blank = {charset.num_classes} classes")
    print(f"  most common train chars: {train_chars.most_common(10)}")
    unused = [c for c in charset.chars if c not in train_chars]
    print(f"  charset chars never seen in train: {''.join(unused) or '(none)'}")

    ds = datasets[args.split]
    if len(ds) == 0:
        print(f"\n{args.split} split is empty; nothing to plot.")
        return
    widths = [ds[i][0].shape[2] for i in range(min(len(ds), 500))]
    print(f"\n  {args.split} image widths after resize (first {len(widths)}): "
          f"min {min(widths)}, median {int(statistics.median(widths))}, max {max(widths)}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(0))
    batch = next(iter(loader))
    print("\n=== One batch ===")
    print(f"  images         {tuple(batch['images'].shape)}  (B, C, H, W_max)")
    print(f"  widths         {batch['widths'].tolist()}")
    print(f"  targets        {tuple(batch['targets'].shape)}  (all labels concatenated)")
    print(f"  target_lengths {batch['target_lengths'].tolist()}")
    print(f"  texts          {batch['texts']}")
    print(f"  decode check   {charset.decode(batch['targets'][:batch['target_lengths'][0]].tolist())!r}"
          f" == {batch['texts'][0]!r}")

    # Grid: each row is one padded image with its label as the title.
    n = len(batch["texts"])
    fig, axes = plt.subplots(n, 1, figsize=(6, 0.8 * n + 0.5))
    axes = [axes] if n == 1 else axes
    for ax, img, text, w in zip(axes, batch["images"], batch["texts"], batch["widths"]):
        ax.imshow(img[0].numpy(), cmap="gray", vmin=-1, vmax=1, aspect="equal")
        ax.axvline(w.item() - 0.5, color="red", lw=0.8)   # red line = where padding starts
        ax.set_title(f"{text!r}  (w={w.item()})", fontsize=8, loc="left")
        ax.set_xticks([]); ax.set_yticks([])   # keep the frame so padding is visible
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nSaved sample grid to {out.resolve()}")


if __name__ == "__main__":
    main()

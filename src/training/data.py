"""Build the datasets for training/evaluation (shared by the scripts and notebook)."""
import time
from pathlib import Path

from src.config import DataConfig
from src.data.charset import Charset
from src.data.iam import make_datasets


def add_dropped(stats, datasets):
    """Count images dropped while loading (unreadable files) as bad images, so the
    data-filtering table stays correct. Safe to call more than once."""
    seen = set(stats.get("bad_image_ids", []))
    new = [w for ds in datasets.values() for w in getattr(ds, "dropped", []) if w not in seen]
    if new:
        stats["bad_or_missing_image"] = stats.get("bad_or_missing_image", 0) + len(new)
        stats["kept"] = stats.get("kept", 0) - len(new)
        stats["bad_image_ids"] = sorted(seen | set(new))
        print(f"counted {len(new)} unreadable image(s) as bad images: {', '.join(new[:10])}")
    return stats


def load_data(data_root=None, splits=("train", "val", "test"), preload=True, check_images=True,
              split_dir="default"):
    """make_datasets() from step 1 (same filters, Aachen splits, writer table),
    then optionally preload the images of the splits we need into RAM."""
    cfg = DataConfig()
    if data_root:
        cfg.iam_root = Path(data_root)
    if split_dir != "default":
        cfg.split_dir = None if split_dir in (None, "none") else Path(split_dir)
    cfg.check_images = check_images
    charset = Charset()
    datasets, stats = make_datasets(cfg, charset)
    sizes = {s: len(datasets[s]) for s in datasets}
    print(f"words per split: {sizes} | split: {stats.get('split_method')}")
    if preload:
        for s in splits:
            t = time.time()
            datasets[s].preload()        # drops (and lists) any image that can't be decoded
            print(f"  preloaded {s}: {len(datasets[s])} images in {time.time() - t:.0f}s")
    add_dropped(stats, datasets)
    return datasets, charset, stats

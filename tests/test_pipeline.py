"""End-to-end: make a tiny fake IAM, load it, and pull one batch."""
import subprocess
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import DataConfig
from src.data.iam import collate_fn, make_datasets

ROOT = Path(__file__).resolve().parents[1]


def test_fake_iam_end_to_end(tmp_path):
    out = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(out),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "3"],
                   check=True, capture_output=True)
    cfg = DataConfig(iam_root=out, split_dir=None)
    ds, stats = make_datasets(cfg)
    # one of each planted bad sample is dropped
    for key in ("segmentation_err", "chars_outside_charset", "too_long", "bad_or_missing_image"):
        assert stats[key] == 1
    assert all(len(ds[s]) > 0 for s in ("train", "val", "test"))
    batch = next(iter(DataLoader(ds["train"], batch_size=4, collate_fn=collate_fn)))
    assert batch["images"].shape[:3] == (4, 1, 32)
    assert batch["targets"].numel() == batch["target_lengths"].sum()

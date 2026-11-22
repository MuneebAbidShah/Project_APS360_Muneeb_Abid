"""Broken word images (empty, garbage, truncated, known-bad IDs) are dropped and counted, never a crash."""
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.config import ModelConfig
from src.data import iam
from src.data.iam import collate_fn, image_path
from src.training.data import add_dropped, load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def broken_iam(tmp_path, monkeypatch):
    """Fake IAM with 4 broken words: zero-byte, garbage bytes, truncated PNG, and one on the known-bad list."""
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    clean, _, base = load_data(fake, preload=False, check_images=True, split_dir=None)
    ids = sorted(it.word_id for d in clean.values() for it in d.items)   # words that pass every filter
    zero, garbage, trunc, known = ids[3], ids[10], ids[17], ids[25]
    image_path(fake, zero).write_bytes(b"")
    image_path(fake, garbage).write_bytes(b"this is not a png" * 20)
    p = image_path(fake, trunc)
    p.write_bytes(p.read_bytes()[:60])        # valid PNG header, pixel data cut off
    monkeypatch.setattr(iam, "KNOWN_BAD_WORD_IDS", iam.KNOWN_BAD_WORD_IDS | {known})
    return fake, {"zero": zero, "garbage": garbage, "trunc": trunc, "known": known}, base


def all_ids(datasets):
    return {it.word_id for ds in datasets.values() for it in ds.items}


def test_known_bad_list_has_the_iam_files():
    assert {"a01-117-05-02", "r06-022-03-05"} <= iam.KNOWN_BAD_WORD_IDS


def test_cheap_path_with_preload_drops_and_counts(broken_iam):
    fake, bad, base = broken_iam
    ds, cs, stats = load_data(fake, preload=True, check_images=False, split_dir=None)
    assert not set(bad.values()) & all_ids(ds)
    assert stats["bad_or_missing_image"] == base["bad_or_missing_image"] + 4
    assert set(stats["bad_image_ids"]) == set(bad.values()) | set(base["bad_image_ids"])
    assert stats["kept"] == sum(len(d) for d in ds.values())
    # zero-byte + known-bad are caught by the cheap file check; garbage/truncated while preloading
    dropped = [w for d in ds.values() for w in d.dropped]
    assert sorted(dropped) == sorted([bad["garbage"], bad["trunc"]])
    n = stats["bad_or_missing_image"]
    add_dropped(stats, ds)
    assert stats["bad_or_missing_image"] == n            # counting again changes nothing


def test_check_images_true_drops_all_upfront(broken_iam):
    fake, bad, base = broken_iam
    ds, _, stats = load_data(fake, preload=False, check_images=True, split_dir=None)
    assert not set(bad.values()) & all_ids(ds) and stats["bad_or_missing_image"] == base["bad_or_missing_image"] + 4


def test_no_preload_skips_unreadable_in_getitem_and_eval(broken_iam):
    from src.models.crnn import CRNN
    from src.report.predict import predict_dataset
    from src.training.engine import evaluate, make_loader
    fake, bad, base = broken_iam
    ds, cs, stats = load_data(fake, preload=False, check_images=False, split_dir=None)
    assert stats["bad_or_missing_image"] == base["bad_or_missing_image"] + 2   # cheap check: zero-byte + known-bad
    model = CRNN(cs.num_classes, ModelConfig(lstm_hidden=16, cnn_channels=(4, 8, 8, 8, 8)))
    n_seen, n_items = 0, 0
    for name, d in ds.items():
        loader, _ = make_loader(d, 4, shuffle=False)
        for batch in loader:                                 # never raises
            if batch is not None:
                n_seen += len(batch["texts"])
        r = evaluate(model, loader, cs, torch.device("cpu"))
        rows = predict_dataset(model, d, cs, torch.device("cpu"), batch_size=4)
        assert r["n"] == len(rows) == len(d) - len(d.skipped)
        assert not {row["word_id"] for row in rows} & set(bad.values())
        n_items += len(d)
    skipped = {w for d in ds.values() for w in d.skipped}
    assert skipped == {bad["garbage"], bad["trunc"]} and n_seen == n_items - 2


def test_collate_drops_none_and_all_none_batch():
    x = torch.ones(1, 32, 20)
    b = collate_fn([None, (x, torch.tensor([1, 2]), 2, "ab"), None])
    assert b["texts"] == ["ab"] and b["images"].shape[0] == 1
    assert collate_fn([None, None]) is None


def test_tesseract_skips_unreadable(broken_iam, tmp_path):
    from src.baselines import tesseract as tess
    fake, bad, base = broken_iam
    for k in ("zero", "garbage", "trunc"):
        assert tess.load_image(image_path(fake, bad[k])) is None
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    pytest.importorskip("pytesseract")
    assert tess.recognize(image_path(fake, bad["garbage"])) is None
    ds, _, _ = load_data(fake, preload=False, check_images=False, split_dir=None)
    items = [it for it in ds["test"].items if it.word_id not in bad.values()][:6]
    g = next(it for d in ds.values() for it in d.items if it.word_id == bad["garbage"])
    rows, _ = tess.run_cached(items + [g], tmp_path / "t.csv", workers=2)
    assert len(rows) == len(items) and bad["garbage"] not in (tmp_path / "t.csv").read_text()

"""Model B's camera-style augmentation: same shape/dtype, train split only, reproducible."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import ModelConfig, TrainConfig
from src.data import augment as A
from src.data.iam import collate_fn
from src.training.data import load_data
from src.training.engine import fit, make_loader

ROOT = Path(__file__).resolve().parents[1]
ALL_ON = A.AugConfig(**{k: 1.0 for k in A.AugConfig().to_dict() if k.startswith("p_")})


def word_image(h=64, w=180, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 235, np.uint8)
    for x in range(10, w - 10, 14):                              # pen-like strokes
        y0 = int(rng.integers(10, h // 2)); img[y0:y0 + h // 3, x:x + 3] = 30
    return img


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    fake = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    ds, cs, stats = load_data(fake, splits=("val", "test"), preload=True, split_dir=None)
    return ds, cs, stats


@pytest.mark.parametrize("shape", [(64, 180), (64, 32), (32, 500), (5, 7)])
def test_shape_and_dtype_preserved(shape):
    aug = A.CameraAugment(ALL_ON)
    img = word_image(*shape) if min(shape) > 8 else np.full(shape, 200, np.uint8)
    for s in range(20):
        out = aug(img, np.random.default_rng(s))
        assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(aug(word_image(), np.random.default_rng(1)), word_image())   # it does something


@pytest.mark.parametrize("op", ["perspective_warp", "rotate", "shear", "motion_blur", "gaussian_blur",
                                "uneven_lighting", "shadow", "contrast", "noise", "jpeg"])
def test_each_distortion_keeps_the_size(op):
    img, rng = word_image(), np.random.default_rng(0)
    args = {"perspective_warp": (0.12,), "rotate": (5.0,), "shear": (0.25,), "motion_blur": (7,),
            "noise": (10.0,), "jpeg": ((25, 80),)}.get(op, ())
    out = getattr(A, op)(img, rng, *args)
    assert out.shape == img.shape
    assert A._u8(out).dtype == np.uint8


def test_deterministic_with_seed():
    aug, img = A.CameraAugment(), word_image()
    a = aug(img, np.random.default_rng([7, 3, 11]))
    b = aug(img, np.random.default_rng([7, 3, 11]))
    c = aug(img, np.random.default_rng([8, 3, 11]))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.array_equal(img, word_image())                     # input never modified in place


def test_wrapper_augments_only_in_train_mode_and_is_reproducible(fake_data):
    ds, cs, _ = fake_data
    tr = A.AugmentedWords(ds["train"], seed=5)
    assert len(tr) == len(ds["train"]) and [i.word_id for i in tr.items] == [i.word_id for i in ds["train"].items]
    x, t, n, text = tr[0]
    assert x.dtype == torch.float32 and x.shape[0] == 1 and x.shape[1] == 32 and text == ds["train"].items[0].text
    clean = tr.image(0, augment=False)
    differs = sum(not np.array_equal(tr.image(i), tr.image(i, augment=False)) for i in range(len(tr)))
    assert differs > len(tr) // 2                                 # most training words get distorted
    off = A.AugmentedWords(ds["train"], seed=5, train=False)
    assert all(np.array_equal(off.image(i), off.image(i, augment=False)) for i in range(len(off)))
    assert torch.equal(off[0][0], off[0][0]) and np.array_equal(off.image(0), clean)
    again = A.AugmentedWords(ds["train"], seed=5)                 # same seed -> same images
    assert all(torch.equal(tr[i][0], again[i][0]) for i in range(5))
    tr.set_epoch(2)                                                # new epoch -> new distortions
    assert tr.epoch == 2 and any(not torch.equal(tr[i][0], again[i][0]) for i in range(5))


def test_val_and_test_stay_clean(fake_data):
    ds, cs, _ = fake_data
    A.AugmentedWords(ds["train"], seed=0)
    for s in ("val", "test"):
        assert ds[s].transform.augment is None and not isinstance(ds[s], A.AugmentedWords)
        x1, x2 = ds[s][0][0], ds[s][0][0]
        assert torch.equal(x1, x2)                                 # no randomness at all


def test_widths_match_the_samples(fake_data):
    ds, cs, _ = fake_data
    tr = A.AugmentedWords(ds["train"], seed=0)
    w = tr.widths()
    assert all(tr[i][0].shape[2] == w[i] for i in range(len(tr)))


def test_same_batches_with_dataloader_workers(fake_data):
    ds, cs, _ = fake_data
    tr = A.AugmentedWords(ds["train"], seed=1)
    tr.set_epoch(3)
    ref = collate_fn([tr[i] for i in range(len(tr))])
    loader = torch.utils.data.DataLoader(tr, batch_size=len(tr), collate_fn=collate_fn, num_workers=2)
    got = next(iter(loader))
    assert torch.equal(ref["images"], got["images"])               # workers see the epoch and seed


def test_unreadable_training_image_is_dropped(fake_data, tmp_path):
    ds, cs, _ = fake_data
    base = ds["train"]
    bad = tmp_path / "bad.png"; bad.write_bytes(b"not a png")
    import copy
    b2 = copy.copy(base); b2.items = list(base.items); b2.dropped = []
    b2.items[0] = copy.copy(b2.items[0]); b2.items[0].path = bad
    tr = A.AugmentedWords(b2, seed=0)
    assert len(tr) == len(base) - 1 and tr.dropped == [base.items[0].word_id]


def test_fit_trains_with_augmentation(fake_data, tmp_path):
    ds, cs, _ = fake_data
    tr = A.AugmentedWords(ds["train"], seed=0)
    tc = TrainConfig(epochs=2, batch_size=8, num_workers=0, out_dir=tmp_path / "run", max_batches=3)
    r = fit({"train": tr, "val": ds["val"]}, cs, ModelConfig(lstm_hidden=32, cnn_channels=(8, 16, 16, 32, 32)),
            tc, verbose=False)
    assert len(r["history"]) == 2 and tr.epoch == 2 and (tmp_path / "run" / "best.pt").exists()


def test_model_b_report_outputs(fake_data, tmp_path):
    import json
    from src.report import model_b_report as mb
    ds, cs, _ = fake_data
    tr = A.AugmentedWords(ds["train"], seed=0)
    out = tmp_path / "model_b"
    figs = mb.save_augmentation_examples(out, tr)
    assert all(Path(p).exists() for p in figs) and len(figs) == 4
    hist = tmp_path / "history.csv"
    hist.write_text("epoch,train_loss,val_loss,val_cer,val_wer,lr,secs\n1,3.0,2.9,0.9,1.0,0.001,10\n"
                    "2,2.0,2.1,0.5,0.8,0.001,10\n")
    curves = mb.save_curves(out, hist)
    assert all(Path(p).name.startswith("fig_model_b_") and Path(p).exists() for p in curves)
    preds = {s: [{"ref": "the", "hyp": "tha"}, {"ref": "cat", "hyp": "cat"}] for s in ("val", "test")}
    ev = {s: {"loss": 1.0, "ms_per_word": 0.5} for s in ("val", "test")}
    a = tmp_path / "runA"; a.mkdir()
    (a / "metrics.json").write_text(json.dumps({"val": {"cer": 0.1, "wer": 0.2}, "test": {"cer": 0.12, "wer": 0.25}}))
    nums, paths = mb.save_metrics(out, preds, ev, {"lr": 1e-3}, "runs/crnn_aug_x", A.AugConfig().to_dict(),
                                  mb.load_model_a_metrics(a))
    assert abs(nums["test"]["cer"] - 1 / 6) < 1e-9 and (out / "table_model_a_vs_b.tex").exists()
    assert mb.load_model_a_metrics(tmp_path / "missing") is None

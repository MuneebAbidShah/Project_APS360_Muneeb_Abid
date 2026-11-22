"""EMNIST baseline: data download/reading/orientation, the CNN and training (resume), the letter
splitter and normalization, decoding with the prior, the IAM run (bad images skipped), report + comparison."""
import csv
import gzip
import json
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.baselines import char_segment as cs
from src.baselines import emnist as E
from src.baselines import emnist_pipeline as P

ROOT = Path(__file__).resolve().parents[1]


def write_idx(path, arr):
    arr = np.asarray(arr, np.uint8)
    head = struct.pack(">HBB", 0, 8, arr.ndim) + struct.pack(">" + "I" * arr.ndim, *arr.shape)
    with gzip.open(path, "wb") as f:
        f.write(head + arr.tobytes())


def fake_emnist(n, seed=0):
    """Tiny EMNIST-like set: label 1 = vertical bar, others random blobs; stored TRANSPOSED like EMNIST."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, len(E.CLASSES), n).astype(np.uint8)
    y[: n // 4] = 1
    x = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        if y[i] == 1:
            x[i, 4:24, 12:16] = 255
        else:
            cv2.circle(x[i], (14, 14), int(rng.integers(4, 10)), 255, 2)
    return x.transpose(0, 2, 1).copy(), y


def test_read_idx_and_orientation(tmp_path):
    x, y = fake_emnist(40)
    write_idx(tmp_path / "x.gz", x); write_idx(tmp_path / "y.gz", y)
    assert (E.read_idx(tmp_path / "x.gz") == x).all() and (E.read_idx(tmp_path / "y.gz") == y).all()
    up, tr = E.orient(x, y)
    assert tr and up[y == 1][0, 4:24, 12:16].min() == 255          # bar is vertical again
    again, tr2 = E.orient(up, y)
    assert not tr2 and (again == up).all()                         # already upright: left alone


def test_download_falls_back_and_reports_failure(tmp_path, monkeypatch):
    x, y = fake_emnist(12)
    monkeypatch.setattr(E, "COUNTS", {"train": 12, "test": 12})
    def fake_fetch(url, dest, timeout=60, retries=3):
        if "bad" in url:
            return False
        write_idx(dest, x if "images" in str(dest) else y)
        return True
    monkeypatch.setattr(E, "_fetch", fake_fetch)
    sources = [("broken mirror", "hf", "bad/repo"), ("good mirror", "hf", "good/repo")]
    assert E.download(tmp_path / "d", sources, verbose=False) == "good mirror"
    assert E.download(tmp_path / "d", sources, verbose=False) == "cached"
    with pytest.raises(RuntimeError, match="Could not download EMNIST"):
        E.download(tmp_path / "e", [("broken mirror", "hf", "bad/repo")], verbose=False)
    d = E.load_emnist(tmp_path / "d", verbose=False)
    assert d["transposed"] and d["train_x"].shape == (12, 28, 28)


def test_cnn_augment_and_train_resume(tmp_path):
    m = E.CharCNN()
    assert m(torch.zeros(3, 1, 28, 28)).shape == (3, 62)
    assert E.augment(torch.rand(5, 1, 28, 28)).shape == (5, 1, 28, 28)
    x, y = fake_emnist(300)
    data = {"train_x": E.orient(x, y)[0], "train_y": y, "test_x": E.orient(x, y)[0][:50], "test_y": y[:50]}
    _, info = E.train(data, tmp_path / "run", epochs=1, batch_size=64, device=torch.device("cpu"), verbose=False)
    for f in ("best.pt", "last.pt", "history.csv", "emnist_classifier.json"):
        assert (tmp_path / "run" / f).exists()
    assert len(info["history"]) == 1 and 0 <= info["test"]["accuracy"] <= 1 and info["parameters"] > 100_000
    _, info2 = E.train(data, tmp_path / "run", epochs=1, batch_size=64, device=torch.device("cpu"), verbose=False)
    assert info2["history"] == info["history"]                    # resumed: nothing re-trained
    w = E.class_weights(np.array([0] * 90 + [1] * 10))
    assert w[1] > w[0]


def word_image(n_letters=3, connected=False, dot=False, slant=0.0, stroke=4):
    """Synthetic word: n vertical letter strokes (optionally joined at the bottom), optional period."""
    img = np.full((60, 40 * n_letters + 40), 255, np.uint8)
    for i in range(n_letters):
        x = 20 + 40 * i
        cv2.line(img, (x, 15), (x, 45), 0, stroke); cv2.line(img, (x + 14, 25), (x + 14, 45), 0, stroke)
        cv2.line(img, (x, 25), (x + 14, 25), 0, stroke)
        if connected and i < n_letters - 1:
            cv2.line(img, (x + 14, 45), (x + 40, 45), 0, stroke)
    if dot:
        cv2.circle(img, (40 * n_letters + 12, 44), 3, 0, -1)
    if slant:
        img = cs._shear(img, -slant, 255)
    return img


def test_splitter_separate_connected_punct_and_edge_cases(tmp_path):
    sw = cs._stroke_width(word_image(3) < 128)
    p = cs.SegParams(width_factor=38 / sw)        # expected width ~38 px: the synthetic letters are 40 px apart
    pieces, crops = cs.segment_word(word_image(3), p)
    assert [q["kind"] for q in pieces] == ["char"] * 3 and crops.shape == (3, 28, 28)
    pieces, _ = cs.segment_word(word_image(3, connected=True), p)       # one joined blob -> split
    assert len([q for q in pieces if q["kind"] == "char"]) == 3
    pieces, _ = cs.segment_word(word_image(2, dot=True), p)              # period at the baseline
    assert pieces[-1]["kind"] == "punct" and pieces[-1]["char"] == "."
    dot = np.full((10, 10), 255, np.uint8); cv2.circle(dot, (5, 5), 3, 0, -1)
    pieces, crops = cs.segment_word(dot)
    assert [q["char"] for q in pieces] == ["."] and len(crops) == 0
    dash = np.full((12, 30), 255, np.uint8); dash[5:8, 3:27] = 0
    assert [q["char"] for q in cs.segment_word(dash)[0]] == ["-"]
    assert cs.segment_word(np.full((40, 80), 255, np.uint8))[0] == []    # blank: nothing, no crash
    (tmp_path / "zero.png").write_bytes(b""); (tmp_path / "junk.png").write_bytes(b"not a png" * 9)
    assert cs.load_gray(tmp_path / "zero.png") is None and cs.load_gray(tmp_path / "junk.png") is None


def test_deslant_and_emnist_normalization():
    slanted = word_image(3, slant=0.4) < 128
    _, s = cs.deslant(slanted)
    assert 0.2 <= s <= 0.6
    m = np.zeros((80, 30), bool); m[5:75, 10:14] = True                  # thin tall stroke
    out = cs.to_emnist(m)
    assert out.shape == (28, 28) and out.dtype == np.uint8 and out.max() == 255 and out[0].max() == 0
    ys, xs = np.nonzero(out > 64)
    assert abs(ys.mean() - 13.5) < 2 and abs(xs.mean() - 13.5) < 2 and ys.max() - ys.min() >= 20
    assert (out > 64).sum(1)[14] >= 2                                     # thickened to EMNIST-like strokes


def test_decode_with_prior_and_punct():
    pieces = [{"kind": "char"}, {"kind": "punct", "char": "'"}, {"kind": "char"}]
    lp = np.full((2, 62), -10.0); lp[:, E.CLASSES.index("O")] = -1.0; lp[:, E.CLASSES.index("o")] = -1.2
    assert cs.decode(pieces, lp) == "O'O"
    prior = cs.char_prior(["foo", "too", "moon"], np.array([E.CLASSES.index("O")] * 50 + [E.CLASSES.index("o")] * 5))
    assert cs.decode(pieces, lp, prior, 1.0) == "o'o"                     # IAM text is mostly lower case
    assert cs.decode([], np.zeros((0, 62))) == ""


@pytest.fixture
def fake_iam(tmp_path):
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"], check=True, capture_output=True)
    from src.training.data import load_data
    ds, _, _ = load_data(fake, splits=(), preload=False, check_images=False, split_dir=None)
    return ds


def test_pipeline_on_fake_iam_skips_bad_images(fake_iam, tmp_path):
    ds = fake_iam
    items = ds["test"].items
    bad = items[1]
    Path(bad.path).write_bytes(b"garbage" * 10)
    m, dev = E.CharCNN().eval(), torch.device("cpu")
    prior = cs.char_prior([it.text for it in ds["train"].items], np.arange(62).repeat(3))
    best, alpha, table = P.tune(ds["val"].items, m, dev, prior, width_factors=(6.0, 8.0))
    assert len(table) == 4 and table[0]["cer"] <= table[-1]["cer"] and best.width_factor in (6.0, 8.0)
    rows, info = P.run(items, m, dev, best, prior, alpha, progress=False)
    assert info["skipped"] == [bad.word_id] and len(rows) == len(items) - 1
    assert bad.word_id not in {r["word_id"] for r in rows}
    s = P.scores(rows)
    assert {"cer", "wer", "cer_case_insensitive", "words"} <= set(s) and info["ms_per_word"] > 0
    path = P.save_rows(rows, tmp_path / "out" / "emnist_test_predictions.csv")
    assert set(next(csv.DictReader(open(path)))) == set(P.FIELDS)


def test_emnist_report_and_three_way_comparison(fake_iam, tmp_path):
    from src.report import compare, emnist_report as er
    from src.report.manifest import Report, future_work, headlines
    ds = fake_iam
    rep = Report(tmp_path / "report", 4)
    assert any("EMNIST" in f for f in future_work(rep))
    m, dev = E.CharCNN().eval(), torch.device("cpu")
    preds, infos = {}, {}
    for split in ("val", "test"):
        preds[split], infos[split] = P.run(ds[split].items, m, dev, cs.SegParams(), None, 0.0, progress=False)
        P.save_rows(preds[split], rep.dir / f"emnist_{split}_predictions.csv")
    clf = {"test": {"accuracy": 0.87, "accuracy_case_insensitive": 0.93, "accuracy_digits": 0.99, "accuracy_upper": 0.9,
                    "accuracy_lower": 0.8, "n": 100}, "cfg": {"train_images": 1000}, "parameters": 450270,
           "history": [{"epoch": 1}], "best_epoch": 1, "minutes": 2.0, "device": "cpu"}
    er.save_classifier(rep, clf)
    tuning = [{"width_factor": 8.0, "prior": 1.0, "cer": 0.5, "wer": 0.8}, {"width_factor": 6.0, "prior": 0.0, "cer": 0.7, "wer": 0.9}]
    er.save_metrics(rep, preds, infos, {s: len(ds[s]) for s in preds}, tuning, cs.params_dict(cs.SegParams()), "cpu", clf)
    ex = er.segmentation_examples(ds["test"], infos["test"]["segs"], infos["test"]["logps"], None, 0.0, n=3)
    assert len(er.save_segmentation_figure(rep, ex)) == 2
    er.save_failure_grid(rep, ds["test"], preds["test"])
    assert len(er.save_pipeline(rep)) == 2
    assert compare.save_comparison(rep) is None                         # no CRNN yet
    # fake CRNN (perfect) and Tesseract (on a subset) results -> 3-way table on the common words
    test_ids = [r["word_id"] for r in preds["test"]]
    for prefix, ids in (("crnn", test_ids), ("tesseract", test_ids[:4])):
        with open(rep.dir / f"{prefix}_test_predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["word_id", "writer", "ref", "hyp", "correct"]); w.writeheader()
            for r in preds["test"]:
                if r["word_id"] in ids:
                    w.writerow({"word_id": r["word_id"], "writer": "", "ref": r["ref"], "hyp": r["ref"], "correct": 1})
    (rep.dir / "crnn_metrics.json").write_text(json.dumps({"device": "Tesla T4", "test": {"cer": 0.1, "wer": 0.2,
                                                           "words": len(test_ids), "ms_per_word_batched": 0.14}}))
    (rep.dir / "tesseract_metrics.json").write_text(json.dumps({"test": {"ms_per_word_cpu_core": 80.0}}))
    rows = compare.save_comparison(rep)
    assert [r["Model"] for r in rows] == ["CRNN (ours)", "Tesseract (LSTM)", "EMNIST CNN + splitter"]
    assert rows[0]["Test CER"] == "0.00" and rows[2]["Params"] == "0.45M"
    nums = json.loads((rep.dir / "report_numbers.json").read_text())
    assert nums["comparison"]["test_words"] == 4 and "All models scored on the same 4" in nums["comparison"]["note"]
    assert any(h.startswith("EMNIST baseline") for h in headlines(nums))
    assert not any("EMNIST" in f for f in future_work(rep))              # done once Step 4 has results
    for f in ("fig_emnist_segmentation.pdf", "fig_emnist_failures.png", "fig_emnist_pipeline.pdf",
              "table_emnist_metrics.tex", "table_emnist_classifier.tex", "table_emnist_tuning.tex",
              "fig_cer_by_length_comparison.pdf", "emnist_metrics.json"):
        assert (rep.dir / f).exists(), f
    for p in rep.dir.glob("*.tex"):
        assert p.read_text().isascii(), p.name
    summary = Path(rep.write_summary(future_work(rep))).read_text()
    assert "Step 4 = EMNIST" in summary and "table_emnist_metrics.tex" in summary

"""Step 5: photo clean-up, CPU export, live-demo keys/captures, the bundle, scoring and report files."""
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from webcam_helpers import ROOT, tiny_checkpoint, word_images
from src.demo import live_demo as L
from src.demo.synth import fake_frame
from src.demo.webcam import IAM_PAPER, Recognizer, clean_photo, crop, export_model, guide_box


def test_clean_photo_makes_iam_like_crops():
    g = word_images(["hardly"])[0]
    frame, box = fake_frame(g, np.random.default_rng(0), lines=True)
    raw = crop(frame, box)
    c = clean_photo(raw)
    assert c is not None and c.dtype == np.uint8
    assert c.shape[0] < raw.shape[0] and c.shape[1] < raw.shape[1]        # tight crop around the ink
    assert np.median(c) == IAM_PAPER                                         # plain IAM-like paper
    assert np.percentile(c, 2) < 120                                          # dark ink
    assert (c[0] < 200).any() and (c[:, 0] < 200).any()                     # ink touches the crop (IAM style)
    paper = np.full((120, 400, 3), 190, np.uint8) + np.random.default_rng(1).integers(0, 6, (120, 400, 3)).astype(np.uint8)
    assert clean_photo(paper) is None                                         # empty box -> nothing


def test_clean_photo_keeps_clean_iam_images_almost_unchanged():
    g = word_images(["could"])[0]
    c = clean_photo(g)
    assert c is not None and abs(c.shape[0] - g.shape[0]) <= 14 and c.shape[1] <= g.shape[1]


def test_export_and_recognizer_match_training_transform(tmp_path):
    from src.data.transforms import Preprocess
    from src.training.engine import load_model
    ck = tmp_path / "best.pt"
    tiny_checkpoint(ck)
    out = export_model(ck, tmp_path / "crnn_model.pt", {"run": "test"})
    assert out["format"] == "aps360-crnn-cpu-v1" and "optimizer" not in out
    torch.load(tmp_path / "crnn_model.pt", weights_only=True)                 # plain tensors + settings only
    rec = Recognizer(tmp_path / "crnn_model.pt")
    ref_model, cs, _ = load_model(ck, torch.device("cpu"))
    g = word_images(["the"])[0]
    x = rec.tensor(g)
    assert torch.equal(x, Preprocess()(Image.fromarray(g)))                   # exactly the training transform
    with torch.no_grad():
        assert torch.allclose(rec.model(x[None]), ref_model(x[None]))
    text, model_u8, cleaned = rec.read(g, clean=False)
    assert model_u8.shape[0] == 32 and cleaned.shape == g.shape and set(text) <= set(cs.chars)


def test_keys_word_and_page_modes():
    k = L.Keys()
    assert k.handle(-1) is None and k.view == "word"
    assert k.handle(ord("c")) == "label" and k.mode == "label"
    for ch in "qa":                                    # letters are typed in label mode, not commands
        assert k.handle(ord(ch)) == "edit"
    assert k.handle(L.KEY_BACK) == "edit" and k.text == "q"
    assert k.handle(L.KEY_ENTER) == "save"
    k.saved(); assert k.mode == "live" and k.text == ""
    assert k.handle(ord("c")) == "label" and k.handle(L.KEY_ENTER) == "empty"
    assert k.handle(L.KEY_ESC) == "cancel" and k.mode == "frozen"
    assert k.handle(L.KEY_SPACE) == "unfreeze" and k.handle(L.KEY_SPACE) == "freeze"
    assert k.handle(ord("d")) == "wider"
    assert k.handle(ord("m")) == "mode" and k.view == "page" and k.mode == "live"
    assert [k.handle(ord(c)) for c in "rsn"] == ["reset", "save_text", "next_camera"]
    assert k.handle(ord("c")) == "page_capture" and k.mode == "frozen"
    assert k.handle(ord("d")) is None                  # word-mode keys do nothing in page mode
    assert k.handle(ord("m")) == "mode" and k.view == "word"
    assert k.handle(ord("q")) == "quit"


def test_smoother_majority_and_ties():
    s = L.Smoother(5)
    assert s.value() == ""
    for t in ["cat", "cot", "cat", "cut"]:
        s.add(t)
    assert s.value() == "cat"
    s.add("cot")
    assert s.value() == "cot"                          # tie cat/cot -> the most recent


def test_save_capture_numbering_and_header_upgrade(tmp_path):
    out = tmp_path / "webcam_test"
    img = np.full((40, 120, 3), 200, np.uint8)
    os.makedirs(out)
    (out / "labels.csv").write_text("file,label,prediction,camera,saved_at,frame_w,frame_h\nimages/0001.png,old,,cam,,640,480\n")
    os.makedirs(out / "images"); cv2.imwrite(str(out / "images" / "0001.png"), img)
    rel, n = L.save_capture(str(out), img, " word ", "ward", "camera 1", (480, 640))
    assert rel == "images/0002.png" and n == 2
    rel, n = L.save_capture(str(out), img, "guess", "guess", "camera 1 page", (480, 640), needs_check=True)
    rows = list(csv.DictReader(open(out / "labels.csv")))
    assert [r["label"] for r in rows] == ["old", "word", "guess"]
    assert [r["needs_check"] for r in rows] == ["0", "0", "1"]
    from src.demo.score import read_labels
    unchecked = []
    assert [r["label"] for r in read_labels(str(out), unchecked)] == ["old", "word"] and unchecked == ["images/0003.png"]


def _fake_frame_file(tmp_path, text="could"):
    frame, _ = fake_frame(word_images([text])[0], np.random.default_rng(2))
    p = tmp_path / "frame.png"; cv2.imwrite(str(p), frame)
    return p


def test_bundle_is_self_contained_and_runs_headless(tmp_path):
    from src.demo import bundle
    ck = tmp_path / "best.pt"; tiny_checkpoint(ck)
    export_model(ck, tmp_path / "crnn_model.pt")
    sample = tmp_path / "s.png"; cv2.imwrite(str(sample), word_images(["sea"])[0])
    live = tmp_path / "live_demo"
    files = bundle.build(live, tmp_path / "crnn_model.pt", {"run": "x"}, samples=[(sample, "sea")])
    for f in ("live_demo.py", "README.md", "requirements.txt", "crnn_model.pt", "src/demo/webcam.py",
              "src/demo/page_reader.py", "src/data/transforms.py", "src/models/crnn.py", "samples/s.png"):
        assert f in files
    for f in bundle.BUNDLED:                                          # unchanged copies of the training code
        assert (live / f).read_bytes() == (ROOT / f).read_bytes()
    assert "torch" in (live / "requirements.txt").read_text() and "CAP_DSHOW" in (live / "live_demo.py").read_text()
    (live / "webcam_test").mkdir(); (live / "webcam_test" / "labels.csv").write_text("file,label\n")
    frame = _fake_frame_file(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run = lambda *a: subprocess.run([sys.executable, "live_demo.py", "--image", str(frame), "--headless", *a],
                                    cwd=tmp_path / "live_demo", env=env, capture_output=True, text=True, timeout=120)
    r = run("--max-frames", "4", "--capture-label", "could")
    assert r.returncode == 0, r.stderr
    assert "FINAL:" in r.stdout and (live / "webcam_test" / "images" / "0001.png").exists()
    r = run("--page", "--max-frames", "3")
    assert r.returncode == 0 and "TRANSCRIPT:" in r.stdout, r.stderr
    z = bundle.make_zip(live, tmp_path / "live_demo.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "live_demo/live_demo.py" in names and "live_demo/crnn_model.pt" in names
    assert not any("webcam_test" in n for n in names)                  # the user's captures stay out of the zip


def test_score_webcam_set_and_report(tmp_path):
    from src.baselines.emnist import CharCNN
    from src.baselines import char_segment as cs
    from src.demo import score as S
    from src.report import webcam_report as wr
    from src.report.manifest import Report, headlines
    ck = tmp_path / "best.pt"; tiny_checkpoint(ck)
    export_model(ck, tmp_path / "m.pt")
    rec = Recognizer(tmp_path / "m.pt")
    wt = tmp_path / "webcam_test"
    rng = np.random.default_rng(0)
    texts = ["the", "sea", "could", "means", "small", "of"]
    for t, g in zip(texts, word_images(texts)):
        frame, box = fake_frame(g, rng)
        L.save_capture(str(wt), crop(frame, box), t, "", "fake", frame.shape)
    L.save_capture(str(wt), crop(frame, box), "guess", "guess", "fake page", frame.shape, needs_check=True)
    assert S.find_webcam_dirs(tmp_path) == [str(wt)]
    unchecked = []
    rows, unreadable = S.load_rows(str(wt), unchecked)
    assert len(rows) == 6 and unchecked == ["images/0007.png"] and not unreadable
    hyp, raw, ms = S.run_crnn(rows, rec)
    clf = CharCNN().eval()
    em, _ = S.run_emnist(rows, clf, torch.device("cpu"), cs.SegParams())
    hyps = {"crnn": hyp, "emnist": em, wr.RAW: raw}
    if shutil.which("tesseract"):
        hyps["tesseract"], _ = S.run_tesseract(rows)
    rep = Report(tmp_path / "report", step=5)
    rep.numbers("comparison", {"CRNN (ours)": {"test": {"cer": 0.1075, "wer": 0.258}},
                               "EMNIST CNN + splitter": {"test": {"cer": 0.5422, "wer": 0.768}}})
    wr.save_predictions(rep, rows, hyps)
    nums = wr.save_metrics(rep, rows, hyps, {"crnn": ms}, {"folder": str(wt), "unchecked_skipped": 1})
    assert nums["words"] == 6 and set(nums["models"]) >= {"crnn", "emnist", wr.RAW}
    assert nums["models"]["crnn"]["iam_test_cer"] == 0.1075
    wr.save_vs_iam_figure(rep, nums); wr.save_prediction_grid(rep, rows, hyp); wr.save_preprocessing_figure(rep, rows, rec)
    d = tmp_path / "report"
    for f in ("table_webcam_metrics.csv", "table_webcam_metrics.tex", "fig_webcam_vs_iam.png", "fig_webcam_vs_iam.pdf",
              "fig_webcam_predictions.png", "fig_webcam_preprocessing.pdf", "webcam_predictions.csv", "webcam_metrics.json"):
        assert (d / f).exists(), f
    tex = (d / "table_webcam_metrics.tex").read_text()
    assert "\\toprule" in tex and "\\midrule" in tex and "%" not in tex.replace("\\%", "")
    assert any(h.startswith("Webcam test set (6 words)") for h in headlines(rep.numbers()))
    assert "Step 5" in open(rep.write_summary()).read()


def test_iam_train_texts_for_the_prior(tmp_path):
    from src.demo.score import iam_train_texts
    (tmp_path / "train.uttlist").write_text("a01-000u\n")
    w = tmp_path / "words.txt"
    w.write_text("# comment\na01-000u-00-00 ok 154 1 2 3 4 AT A\na01-000u-00-01 err 154 1 2 3 4 NN MOVE\n"
                 "b01-000-00-00 ok 154 1 2 3 4 NN test\n")
    assert iam_train_texts(w, tmp_path, "AMOVEtes") == ["A"]

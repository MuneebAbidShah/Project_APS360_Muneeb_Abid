"""Step 5 with two CRNNs: Model A and Model B (augmented) scored on the webcam set next to the baselines,
the 4-row webcam table, per-model prediction files, and the same-architecture check for Model B."""
import csv
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from webcam_helpers import tiny_checkpoint, word_images
from src.demo import live_demo as L
from src.demo import score as S
from src.demo.synth import fake_frame
from src.demo.webcam import Recognizer, crop, export_model


def _model_b_checkpoint(path, seed=1, **cfg_changes):
    """Same tiny architecture as tiny_checkpoint (different weights), optionally with changed settings."""
    from src.config import ModelConfig
    from src.data.charset import Charset
    from src.models.crnn import CRNN
    torch.manual_seed(seed)
    cfg = ModelConfig(**{"cnn_channels": (8, 8, 16, 16, 16), "lstm_hidden": 16, **cfg_changes})
    cs = Charset("abcdefghijklmnopqrstuvwxyz")
    m = CRNN(cs.num_classes, cfg).eval()
    torch.save({"model": m.state_dict(), "optimizer": {}, "epoch": 7, "best_cer": 0.4,
                "model_cfg": asdict(cfg), "charset": cs.chars}, path)


@pytest.fixture
def two_models(tmp_path):
    tiny_checkpoint(tmp_path / "a.pt")
    export_model(tmp_path / "a.pt", tmp_path / "crnn_model.pt", {"model": "A"})
    _model_b_checkpoint(tmp_path / "b.pt")
    export_model(tmp_path / "b.pt", tmp_path / "crnn_model_b.pt", {"model": "B"})
    return Recognizer(tmp_path / "crnn_model.pt"), Recognizer(tmp_path / "crnn_model_b.pt")


def test_model_b_has_model_a_architecture(two_models, tmp_path):
    rec_a, rec_b = two_models
    assert S.architecture_diff(rec_a, rec_b) == []
    # weights really are different models (else the comparison would be meaningless)
    sa, sb = rec_a.model.state_dict(), rec_b.model.state_dict()
    assert any(not torch.equal(sa[k], sb[k]) for k in sa)
    # config.json of the training run (strings/lists as written by the trainer) matches the checkpoint
    cfg = {"model": {"cnn_channels": [8, 8, 16, 16, 16], "lstm_hidden": 16, "lstm_layers": 2, "dropout": 0.3}}
    assert S.config_diff(rec_b, cfg) == []
    assert S.config_diff(rec_b, {"lstm_hidden": 256}) == ["lstm_hidden: checkpoint=16 config.json=256"]


def test_architecture_diff_catches_mismatches(two_models, tmp_path):
    rec_a, _ = two_models
    _model_b_checkpoint(tmp_path / "wide.pt", lstm_hidden=32)
    diffs = S.architecture_diff(rec_a, Recognizer(tmp_path / "wide.pt"))
    assert diffs == ["model_cfg.lstm_hidden: A=16 B=32"]
    tiny_checkpoint(tmp_path / "abc.pt", chars="abc")
    assert "charset differs" in S.architecture_diff(rec_a, Recognizer(tmp_path / "abc.pt"))
    rec_c = Recognizer(tmp_path / "abc.pt"); rec_c.preprocess_cfg = {**rec_a.preprocess_cfg, "height": 48}
    assert any(d.startswith("preprocess.height") for d in S.architecture_diff(rec_a, rec_c))


def test_both_models_scored_with_baselines(two_models, tmp_path):
    from src.baselines import char_segment as cs
    from src.baselines.emnist import CharCNN
    from src.report import webcam_report as wr
    from src.report.manifest import Report, future_work
    rec_a, rec_b = two_models
    wt = tmp_path / "webcam_test"
    rng = np.random.default_rng(0)
    texts = ["the", "sea", "could", "means", "small"]
    for t, g in zip(texts, word_images(texts)):
        frame, box = fake_frame(g, rng)
        L.save_capture(str(wt), crop(frame, box), t, "", "fake", frame.shape)
    L.save_capture(str(wt), crop(frame, box), "guess", "guess", "fake page", frame.shape, needs_check=True)
    # a labels.csv row whose image was deleted is skipped
    with open(wt / "labels.csv", "a", newline="") as f:
        csv.writer(f).writerow(["images/0099.png", "gone", "", "fake", "", 1, 1, 0])
    unchecked = []
    rows, unreadable = S.load_rows(str(wt), unchecked)
    assert [r["label"] for r in rows] == texts and unchecked == ["images/0006.png"] and not unreadable

    hyps, ms = {}, {}
    hyps["crnn"], hyps[wr.RAW], ms["crnn"] = S.run_crnn(rows, rec_a)
    hyps["crnn_b"], raw_b, ms["crnn_b"] = S.run_crnn(rows, rec_b, with_raw=False)
    assert raw_b is None and len(hyps["crnn_b"]) == len(rows)
    hyps["emnist"], ms["emnist"] = S.run_emnist(rows, CharCNN().eval(), torch.device("cpu"), cs.SegParams())
    hyps["tesseract"] = ["the", "", "cold", "means", "x"]          # stand-in so the test needs no tesseract binary

    rep = Report(tmp_path / "outputs" / "report", step=5)
    rep.numbers("comparison", {"CRNN (ours)": {"test": {"cer": 0.1075, "wer": 0.2579}},
                               "Tesseract (LSTM)": {"test": {"cer": 0.5649, "wer": 0.8394}},
                               "EMNIST CNN + splitter": {"test": {"cer": 0.5422, "wer": 0.768}}})
    wr.save_predictions(rep, rows, hyps)
    nums = wr.save_metrics(rep, rows, hyps, ms, {"folder": str(wt)}, iam_cer={"crnn_b": 0.0877})
    d = rep.dir
    with open(d / "table_webcam_metrics.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert [r["Model"] for r in table] == ["Tesseract (LSTM)", "EMNIST CNN + splitter", "CRNN, Model A",
                                          "CRNN + augmentation, Model B"]
    assert table[3]["IAM test CER"] == "8.77" and table[2]["IAM test CER"] == "10.75"
    assert table[0]["CER"] == f"{100 * S.scores(texts, hyps['tesseract'])['cer']:.2f}"
    tex = (d / "table_webcam_metrics.tex").read_text()
    assert tex.count(r"\\") >= 5 and "Model B" in tex and r"\toprule" in tex
    assert (d / "table_webcam_cleanup_ablation.csv").exists()
    for k in ("crnn", "crnn_b", "emnist", "tesseract", wr.RAW):
        with open(d / f"webcam_predictions_{k}.csv", newline="") as f:
            pr = list(csv.DictReader(f))
        assert [r["label"] for r in pr] == texts and [r["prediction"] for r in pr] == hyps[k]
        assert all(r["correct"] == str(int(r["label"] == r["prediction"])) for r in pr)
    assert set(nums["models"]) == {"crnn", "crnn_b", "emnist", "tesseract", wr.RAW}
    assert nums["models"]["crnn_b"]["iam_test_cer"] == 0.0877
    saved = json.loads((d / "webcam_metrics.json").read_text())
    assert saved["models"]["crnn_b"]["name"] == "CRNN + augmentation, Model B"
    wr.save_vs_iam_figure(rep, nums)
    wr.save_prediction_grid(rep, rows, hyps["crnn_b"], name="fig_webcam_predictions_model_b", model="Model B")
    assert (d / "fig_webcam_vs_iam.png").exists() and (d / "fig_webcam_predictions_model_b.pdf").exists()
    # augmentation is no longer "future work" once Step 6 has saved Model B's metrics
    assert any("augmentation" in w.lower() for w in future_work(rep))
    (d.parent / "model_b").mkdir(); (d.parent / "model_b" / "model_b_metrics.json").write_text("{}")
    assert not any("augmentation" in w.lower() for w in future_work(rep))


def test_three_rows_without_model_b(two_models, tmp_path):
    """Model B missing (Step 6 not run): the table simply has the other rows; nothing crashes."""
    from src.report import webcam_report as wr
    from src.report.manifest import Report
    rec_a, _ = two_models
    rows = [{"file": "a.png", "label": "sea", "clean": word_images(["sea"])[0], "raw": None}]
    hyp, _, ms = S.run_crnn(rows, rec_a, with_raw=False)
    rep = Report(tmp_path / "report", step=5)
    nums = wr.save_metrics(rep, rows, {"crnn": hyp, "tesseract": ["sea"]}, {"crnn": ms}, {})
    assert list(nums["models"]) == ["tesseract", "crnn"]
    assert not (rep.dir / "table_webcam_cleanup_ablation.csv").exists()

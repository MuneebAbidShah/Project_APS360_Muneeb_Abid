"""Report outputs: LaTeX tables, error analysis, model summary, figures, numbers, comparison."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from src.config import ModelConfig, TrainConfig
from src.models.crnn import CRNN, count_parameters
from src.report import compare, crnn_report, errors, figures, tables
from src.report.manifest import Report
from src.report.model_summary import layer_table, macs_per_word
from src.report.predict import predict_dataset, save_predictions
from src.training.data import load_data
from src.training.engine import evaluate, make_loader
from src.utils.metrics import edit_distance

ROOT = Path(__file__).resolve().parents[1]
SMALL = ModelConfig(lstm_hidden=32, cnn_channels=(8, 16, 16, 32, 32))


def test_latex_escape_and_table(tmp_path):
    assert tables.latex_escape("a_b & 50% #1 $x") == r"a\_b \& 50\% \#1 \$x"
    assert r"\ensuremath{\times}" in tables.latex_escape("1×32") and "→" not in tables.latex_escape("a→b")
    rows = [{"Split": "val", "CER": 0.1234}, "midrule", {"Split": "te_st", "CER": 0.5}]
    csv_p, tex_p = tables.write_table(rows, ["Split", "CER"], tmp_path, "t", "Cap with words_recovered.txt")
    tex = Path(tex_p).read_text()
    for s in (r"\toprule", r"\midrule", r"\bottomrule", r"\label{tab:t}", r"te\_st", r"words\_recovered.txt", "0.1234"):
        assert s in tex
    assert tex.isascii(), "pdfLaTeX needs ASCII (unicode is converted to commands)"
    assert list(csv.DictReader(open(csv_p)))[1]["Split"] == "te_st"


def test_alignment_matches_edit_distance():
    pairs = [("hello", "helo"), ("the", "thee"), ("of", "to"), ("word", ""), ("", "x"), ("abc", "abc")]
    for r, h in pairs:
        ops = errors.align(r, h)
        assert sum(op != "=" for op, _, _ in ops) == edit_distance(r, h)
        assert "".join(a for op, a, _ in ops if op != "ins") == r
        assert "".join(b for op, _, b in ops if op != "del") == h
    c = errors.char_confusions(["hello", "of"], ["helo", "to"])
    assert c[("l", errors.EMPTY)] == 1 and sum(c.values()) == 3
    bd = errors.error_breakdown(["hello", "of"], ["helo", "to"])
    assert bd["deletions"] + bd["substitutions"] + bd["insertions"] == 3
    rows = errors.cer_by_length(["a", "to", "word", "longerwordhere"], ["a", "tx", "word", "longer"])
    assert sum(r["n"] for r in rows) == 4 and rows[0]["cer"] == 0


def test_model_summary_adds_up():
    for cfg in (ModelConfig(), SMALL):
        model = CRNN(79, cfg)
        rows, total = layer_table(model)
        assert total == count_parameters(model)
        assert rows[0]["output_shape"] == "1×32×W" and rows[-1]["output_shape"] == "T×79"
        assert rows[5]["output_shape"].endswith("×1×W/4")
        assert macs_per_word(model) > 0
    assert count_parameters(CRNN(79)) == 5_335_567


def _history(path, n=6):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr", "secs"])
        w.writeheader()
        for e in range(1, n + 1):
            w.writerow({"epoch": e, "train_loss": 5 / e, "val_loss": 4 / e + 0.1, "val_cer": 0.9 / e,
                        "val_wer": min(1, 1.5 / e), "lr": 1e-3, "secs": 10})


def test_figures_png_300dpi_and_pdf(tmp_path):
    _history(tmp_path / "history.csv")
    paths = figures.plot_training_curves(tmp_path / "history.csv", tmp_path)
    img = [(255 * (i % 2)) * __import__("numpy").ones((40, 90), dtype="uint8") for i in range(4)]
    paths += figures.plot_prediction_grid(img, ["a", "b&$", "c", "d"], ["a", "x", "", "d"], tmp_path, "grid")
    paths += figures.plot_char_confusions(errors.char_confusions(["hello"], ["hallo"]), tmp_path)
    paths += figures.plot_label_length_hist(["a", "to", "word"], tmp_path)
    paths += figures.plot_cer_by_length({"A": errors.cer_by_length(["ab"], ["a"]),
                                         "B": errors.cer_by_length(["ab"], ["ab"])}, tmp_path)
    paths += figures.draw_crnn_diagram(ModelConfig(), 79, tmp_path)
    paths += figures.draw_tesseract_pipeline(tmp_path)
    assert len(paths) == 2 * 9
    for p in paths:
        assert Path(p).stat().st_size > 1000
        if p.endswith(".png"):
            assert round(Image.open(p).info["dpi"][0]) == 300


def _fake(tmp_path):
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    return load_data(fake, splits=("train", "val", "test"), preload=True, split_dir=None, check_images=True)


def test_crnn_report_end_to_end(tmp_path):
    """Untrained small CRNN on fake data -> every Step 2 report file, numbers and summary."""
    ds, cs, stats = _fake(tmp_path)
    model = CRNN(cs.num_classes, SMALL).eval()
    import torch
    dev = torch.device("cpu")
    rep = Report(tmp_path / "report", 2)
    preds, res = {}, {}
    for s in ("val", "test"):
        loader, _ = make_loader(ds[s], 128, shuffle=False)
        res[s] = evaluate(model, loader, cs, dev)
        preds[s] = predict_dataset(model, ds[s], cs, dev)
        assert [p["ref"] for p in preds[s]] == [it.text for it in ds[s].items]
        assert sorted(p["hyp"] for p in preds[s]) == sorted(res[s]["hyps"])   # same predictions as evaluate()
        save_predictions(preds[s], rep.dir / f"crnn_{s}_predictions.csv")
    _history(tmp_path / "history.csv")
    _, hist = crnn_report.save_curves(rep, tmp_path / "history.csv")
    crnn_report.save_metrics(rep, preds, res, 3.0, "cpu")
    crnn_report.save_model_summary(rep, model, cs, {"epoch": 4, "best_cer": 0.2}, TrainConfig(), SMALL, hist, "cpu")
    crnn_report.save_data_stats(rep, stats, ds, "fake")
    crnn_report.save_label_hist(rep, ds)
    assert len(crnn_report.save_preprocessing_examples(rep, ds["train"])) == 2
    crnn_report.save_prediction_grid(rep, ds["test"], preds["test"])
    crnn_report.save_error_analysis(rep, preds["test"])
    crnn_report.save_cer_by_length(rep, preds["test"])
    crnn_report.save_architecture(rep, SMALL, cs.num_classes)
    assert crnn_report.save_sweep(rep, tmp_path / "missing.csv") is None
    assert compare.save_comparison(rep) is None                 # no Tesseract results yet

    # Tesseract results on a subset -> comparison is scored on the same words
    sub = preds["test"][:5]
    with open(rep.dir / "tesseract_test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["word_id", "writer", "ref", "hyp", "correct", "ms"]); w.writeheader()
        w.writerows({**r, "hyp": r["ref"], "correct": 1, "ms": 50} for r in sub)
    (rep.dir / "tesseract_metrics.json").write_text(json.dumps({"test": {"ms_per_word_cpu_core": 80.0}}))
    rows = compare.save_comparison(rep)
    assert rows[1]["Test CER"] == "0.00" and (rep.dir / "table_comparison.tex").exists()
    nums = json.loads((rep.dir / "report_numbers.json").read_text())
    assert nums["comparison"]["CRNN (ours)"]["test"]["words"] == 5
    for sec in ("crnn", "model", "training", "data", "comparison"):
        assert sec in nums
    assert nums["model"]["parameters"] == count_parameters(model)
    summary = Path(rep.write_summary(["EMNIST"])).read_text()
    for name in ("fig_training_curves.pdf", "table_model_summary.tex", "table_comparison.tex", "EMNIST"):
        assert name in summary
    for p in rep.dir.glob("*.tex"):
        assert p.read_text().isascii(), p.name


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tables_compile_with_pdflatex(tmp_path):
    rows = [{"Word": w, "n": 1} for w in ['"', "&", "#", "_", "%", "$", "{", "}", "~", "^", "\\", "<", "|", "'", "`"]]
    tables.write_table(rows, ["Word", "n"], tmp_path, "t", "Caption ×→∅ 50% a_b")
    (tmp_path / "doc.tex").write_text("\\documentclass{article}\\usepackage{booktabs}\\begin{document}"
                                      "\\input{t}\\end{document}")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]

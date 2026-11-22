"""End-to-end smoke test: fake IAM -> fit 2 epochs -> checkpoint -> resume -> evaluate."""
import json
import subprocess
import sys
from pathlib import Path

from src.config import ModelConfig, TrainConfig
from src.training.data import load_data
from src.training.engine import evaluate, fit, load_model, make_loader

ROOT = Path(__file__).resolve().parents[1]
SMALL = ModelConfig(lstm_hidden=32, cnn_channels=(8, 16, 16, 32, 32))


def test_fit_resume_evaluate(tmp_path):
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    ds, cs, _ = load_data(fake, preload=True, split_dir=None)
    out = tmp_path / "run"
    tc = TrainConfig(epochs=2, batch_size=8, num_workers=0, out_dir=out, max_batches=3)
    r = fit(ds, cs, SMALL, tc, verbose=False)
    assert len(r["history"]) == 2
    for f in ("best.pt", "last.pt", "history.csv", "config.json"):
        assert (out / f).exists()

    tc.epochs = 3                                   # resume: only epoch 3 should run
    r2 = fit(ds, cs, SMALL, tc, verbose=False)
    assert [h["epoch"] for h in r2["history"]] == [1, 2, 3]

    model, cs2, ck = load_model(out / "best.pt")
    loader, _ = make_loader(ds["test"], 16, shuffle=False)
    res = evaluate(model, loader, cs2, next(model.parameters()).device)
    assert res["n"] == len(ds["test"]) and 0 <= res["cer"]
    assert json.loads((out / "config.json").read_text())["parameters"] > 0


def test_scripts_run(tmp_path):
    fake = tmp_path / "fake"
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake), "--writers", "10",
                    "--forms-per-writer", "1", "--lines-per-form", "2"], check=True, capture_output=True)
    common = ["--data-root", str(fake), "--split-dir", "none", "--batch-size", "8", "--max-batches", "2",
              "--hidden", "32"]
    r = subprocess.run([py, str(ROOT / "scripts" / "train.py"), *common, "--epochs", "1",
                        "--out-dir", str(tmp_path / "run")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([py, str(ROOT / "scripts" / "evaluate.py"), "--checkpoint", str(tmp_path / "run" / "best.pt"),
                        "--data-root", str(fake), "--split-dir", "none"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "run" / "test_metrics.json").exists()
    r = subprocess.run([py, str(ROOT / "scripts" / "sweep.py"), "--data-root", str(fake), "--split-dir", "none",
                        "--out-dir", str(tmp_path / "sweep"), "--only", "2", "--hidden-scale", "0.125",
                        "--epochs", "1", "--max-batches", "2", "--train-frac", "0.5"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len((tmp_path / "sweep" / "sweep_results.csv").read_text().splitlines()) == 3

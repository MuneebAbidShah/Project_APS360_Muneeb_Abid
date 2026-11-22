"""Sweep design, resumable CSV, winner selection / fallback, and the run-folder guard."""
import csv
import json

import pytest

from src.config import ModelConfig, TrainConfig
from src.data.charset import Charset
from src.training import sweep as sw


def test_design_is_12_unique_runs_around_default():
    cfgs = sw.sweep_configs()
    assert len(cfgs) == 12 and len({sw.run_name(c) for c in cfgs}) == 12
    assert sum(c["group"] == "grid: lr x hidden" for c in cfgs) == 6
    assert {c["lr"] for c in cfgs} == {5e-4, 1e-3, 3e-3}
    assert {c["dropout"] for c in cfgs} == {0.1, 0.3, 0.5}
    assert {c["batch_size"] for c in cfgs} == {32, 64, 128}
    assert {c["lstm_hidden"] for c in cfgs} == {128, 256, 384} and {c["lstm_layers"] for c in cfgs} == {2, 3}
    for c in cfgs:     # one-at-a-time runs change exactly one setting from the default
        if c["group"].startswith("vary"):
            assert sum(c[k] != sw.DEFAULTS[k] for k in sw.SWEPT) == 1
    assert sw.run_name(sw.DEFAULTS) == "lr0.001_h256_d0.3_b64_L2"


class FakeFit:
    """Stands in for engine.fit: records calls and returns a made-up history."""
    def __init__(self, cers):
        self.cers, self.calls = cers, []

    def __call__(self, datasets, charset, mc, tc, resume=True, verbose=False):
        name = sw.run_name({"lr": tc.lr, "lstm_hidden": mc.lstm_hidden, "dropout": mc.dropout,
                            "batch_size": tc.batch_size, "lstm_layers": mc.lstm_layers})
        self.calls.append(name)
        c = self.cers.get(name, 0.5)
        hist = [{"epoch": e, "val_cer": c + 0.1 * (tc.epochs - e), "val_wer": min(1, 2 * c), "secs": 30.0}
                for e in range(1, tc.epochs + 1)]
        return {"history": hist, "best_val_cer": c}


def test_run_sweep_records_everything_and_resumes(tmp_path):
    cfgs = sw.sweep_configs()
    csv_path = tmp_path / "sweep_results.csv"
    best = sw.run_name(cfgs[7])
    fake = FakeFit({best: 0.12})
    # "disconnect" after 5 runs
    sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs[:5], epochs=3, train_frac=0.3, fit_fn=fake, verbose=False)
    assert len(fake.calls) == 5
    rows = list(csv.DictReader(open(csv_path)))
    assert list(rows[0]) == sw.FIELDS and len(rows) == 5
    assert rows[0]["epochs"] == "3" and float(rows[0]["minutes"]) == 1.5 and int(rows[0]["parameters"]) > 0
    # re-run the full sweep: only the 7 missing runs train
    ranked = sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs, epochs=3, train_frac=0.3, fit_fn=fake, verbose=False)
    assert len(fake.calls) == 12 and len(set(fake.calls)) == 12
    assert ranked[0]["name"] == best and len(ranked) == 12
    # a third time: nothing to do
    sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs, epochs=3, train_frac=0.3, fit_fn=fake, verbose=False)
    assert len(fake.calls) == 12
    # different epochs = different experiment, so it runs again
    sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs[:1], epochs=4, train_frac=0.3, fit_fn=fake, verbose=False)
    assert len(fake.calls) == 13


def test_old_format_csv_is_kept_aside(tmp_path):
    csv_path = tmp_path / "sweep_results.csv"
    csv_path.write_text("run,lr,hidden,best_val_cer\nlr0.001_h128,0.001,128,0.5\n")
    assert sw.load_results(csv_path) == []
    sw.run_sweep(None, Charset(), csv_path, tmp_path, sw.sweep_configs()[:1], epochs=1, fit_fn=FakeFit({}), verbose=False)
    assert (tmp_path / "sweep_results_old_format.csv").exists() and len(sw.load_results(csv_path)) == 1


def test_choose_settings_winner_and_fallbacks(tmp_path):
    csv_path = tmp_path / "s.csv"
    s, info = sw.choose_settings(csv_path)                                   # no CSV
    assert s == sw.DEFAULTS and info["source"] == "defaults" and info["winner"] is None
    csv_path.write_text(",".join(sw.FIELDS) + "\n")                          # empty CSV
    assert sw.choose_settings(csv_path)[1]["source"] == "defaults"
    cfgs = sw.sweep_configs()
    cers = {sw.run_name(c): 1.0 for c in cfgs}                                 # nothing learned
    sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs, epochs=1, fit_fn=FakeFit(cers), verbose=False)
    s, info = sw.choose_settings(csv_path)
    assert s == sw.DEFAULTS and "noise" in info["reason"]

    csv2 = tmp_path / "s2.csv"
    a, b = sw.run_name(cfgs[2]), sw.run_name(cfgs[9])
    sw.run_sweep(None, Charset(), csv2, tmp_path, cfgs, epochs=1,
                 fit_fn=FakeFit({**{n: 0.6 for n in cers}, a: 0.2, b: 0.2}), verbose=False)
    s, info = sw.choose_settings(csv2)
    assert info["source"] == "sweep" and info["n_runs"] == 12
    assert info["winner"]["name"] == min(a, b)       # tie on CER and WER -> name decides (deterministic)
    assert s == {k: cfgs[[sw.run_name(c) for c in cfgs].index(min(a, b))][k] for k in sw.SWEPT}
    mc, tc = sw.make_configs(s, epochs=40)
    assert (mc.lstm_hidden, mc.dropout, mc.lstm_layers, tc.lr, tc.batch_size, tc.epochs) == \
           (s["lstm_hidden"], s["dropout"], s["lstm_layers"], s["lr"], s["batch_size"], 40)


def test_check_run_dir_refuses_mismatched_checkpoint(tmp_path):
    from dataclasses import asdict
    mc, tc = sw.make_configs(sw.DEFAULTS, epochs=40)
    assert sw.check_run_dir(tmp_path, mc, tc) == "new run"
    (tmp_path / "config.json").write_text(json.dumps(
        {"model": asdict(mc), "train": {k: str(v) for k, v in asdict(tc).items()}}))
    assert sw.check_run_dir(tmp_path, mc, TrainConfig(**{**asdict(tc), "epochs": 60})) == "same settings"
    with pytest.raises(sw.RunMismatch, match="lstm_hidden"):
        sw.check_run_dir(tmp_path, ModelConfig(lstm_hidden=128), tc)
    with pytest.raises(sw.RunMismatch, match="train.lr"):
        sw.check_run_dir(tmp_path, mc, TrainConfig(**{**asdict(tc), "lr": 3e-3}))


def test_sweep_report_table_marks_chosen_row(tmp_path):
    from src.report.manifest import Report
    from src.report import crnn_report
    csv_path = tmp_path / "s.csv"
    cfgs = sw.sweep_configs()
    win = sw.run_name(cfgs[4])
    sw.run_sweep(None, Charset(), csv_path, tmp_path, cfgs, epochs=2, fit_fn=FakeFit({win: 0.1}), verbose=False)
    rep = Report(tmp_path / "report", 2)
    paths = crnn_report.save_sweep(rep, csv_path, chosen_name=win)
    assert any(p.endswith("fig_sweep.pdf") for p in paths)
    tex = (rep.dir / "table_sweep.tex").read_text()
    assert tex.isascii() and r"\textbf{1*}" in tex and tex.count(r"\textbf{") >= 10 + 10   # header + bold row
    rows = list(csv.DictReader(open(rep.dir / "table_sweep.csv")))
    assert len(rows) == 12 and rows[0]["Chosen"] == "yes" and rows[1]["Chosen"] == ""
    assert {"LR", "Hidden", "Layers", "Dropout", "Batch", "Val CER (%)", "Val WER (%)"} <= set(rows[0])
    nums = json.loads((rep.dir / "report_numbers.json").read_text())
    assert nums["sweep"]["chosen"] == win and len(nums["sweep"]["runs"]) == 12

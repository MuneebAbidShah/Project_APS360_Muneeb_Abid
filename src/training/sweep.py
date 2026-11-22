"""Hyperparameter sweep: short runs ranked by validation CER, and picking the winner.

Design (12 runs, not a full grid):
  * a small grid: learning rate {5e-4, 1e-3, 3e-3} x LSTM hidden {128, 256}   (6 runs)
  * one-at-a-time changes around the default (lr 1e-3, hidden 256, dropout 0.3,
    batch 64, 2 LSTM layers): dropout 0.1 / 0.5, batch 32 / 128, hidden 384,
    3 LSTM layers                                                              (6 runs)
Every run trains for the same number of epochs on the same 30% of the training words
(fixed seed), and is scored on the full validation set. The test set is never used.

Resumable: each finished run is appended to sweep_results.csv straight away, and runs
already in the CSV (same settings, epochs and train fraction) are skipped, so a Colab
disconnect only loses the run in progress (and that one resumes from its last.pt).
"""
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

from src.config import ModelConfig, TrainConfig
from src.models.crnn import CRNN, count_parameters

DEFAULTS = {"lr": 1e-3, "lstm_hidden": 256, "dropout": 0.3, "batch_size": 64, "lstm_layers": 2}
SWEPT = ("lr", "lstm_hidden", "dropout", "batch_size", "lstm_layers")
FIELDS = ["name", "group", *SWEPT, "epochs", "train_frac", "best_val_cer", "best_val_wer", "best_epoch",
          "epochs_run", "minutes", "parameters"]
_TYPES = {"lr": float, "dropout": float, "train_frac": float, "best_val_cer": float, "best_val_wer": float,
          "minutes": float, "lstm_hidden": int, "batch_size": int, "lstm_layers": int, "epochs": int,
          "best_epoch": int, "epochs_run": int, "parameters": int}


def run_name(s: dict) -> str:
    """Short unique name for a setting, e.g. lr0.001_h256_d0.3_b64_L2 (also used in RUN_NAME)."""
    return f"lr{s['lr']:g}_h{s['lstm_hidden']}_d{s['dropout']:g}_b{s['batch_size']}_L{s['lstm_layers']}"


def sweep_configs(defaults=DEFAULTS):
    """The 12 settings described at the top of this file, each with a 'group' label."""
    out = []
    for lr in (5e-4, 1e-3, 3e-3):
        for hidden in (128, 256):
            out.append({**defaults, "lr": lr, "lstm_hidden": hidden, "group": "grid: lr x hidden"})
    for key, values in (("dropout", (0.1, 0.5)), ("batch_size", (32, 128)),
                        ("lstm_hidden", (384,)), ("lstm_layers", (3,))):
        for v in values:
            out.append({**defaults, key: v, "group": f"vary {key}"})
    seen, unique = set(), []
    for c in out:
        if run_name(c) not in seen:
            seen.add(run_name(c)); unique.append(c)
    return unique


def load_results(csv_path):
    """Rows of a sweep_results.csv, with numbers parsed. Rows in an unknown format are dropped."""
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: _TYPES.get(k, str)(r[k]) for k in FIELDS})
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def _legacy_aside(csv_path):
    """A sweep_results.csv from the old 4-run sweep has other columns: keep it, renamed."""
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), [])
        if header and header != FIELDS:
            dst = str(csv_path).replace(".csv", "_old_format.csv")
            shutil.move(csv_path, dst)
            print(f"moved an older-format sweep file to {os.path.basename(dst)} (kept, not used)")


def _done(rows, cfg, epochs, train_frac):
    return any(r["name"] == run_name(cfg) and r["epochs"] == epochs and math.isclose(r["train_frac"], train_frac)
               for r in rows)


def run_sweep(datasets, charset, csv_path, out_root, configs=None, epochs=8, train_frac=0.3,
              max_batches=None, fit_fn=None, verbose=True):
    """Train every config that isn't in `csv_path` yet; append each result as soon as it finishes."""
    from src.training.engine import fit
    fit_fn = fit_fn or fit
    configs = configs or sweep_configs()
    _legacy_aside(csv_path)
    rows = load_results(csv_path)
    todo = [c for c in configs if not _done(rows, c, epochs, train_frac)]
    print(f"sweep: {len(configs)} settings, {len(configs) - len(todo)} already in {os.path.basename(str(csv_path))}, "
          f"{len(todo)} to run ({epochs} epochs on {train_frac:.0%} of train each)")
    t_start = time.time()
    for i, c in enumerate(todo, 1):
        name = run_name(c)
        mc = ModelConfig(lstm_hidden=c["lstm_hidden"], lstm_layers=c["lstm_layers"], dropout=c["dropout"])
        tc = TrainConfig(epochs=epochs, batch_size=c["batch_size"], lr=c["lr"], train_frac=train_frac,
                         patience=epochs, num_workers=0, max_batches=max_batches,
                         out_dir=Path(out_root) / f"{name}_e{epochs}_f{train_frac:g}")
        t0 = time.time()
        r = fit_fn(datasets, charset, mc, tc, resume=True, verbose=False)
        hist = r["history"]
        best = min(hist, key=lambda h: h["val_cer"]) if hist else {"val_cer": math.inf, "val_wer": math.inf, "epoch": 0}
        row = {"name": name, "group": c.get("group", ""), **{k: c[k] for k in SWEPT}, "epochs": epochs,
               "train_frac": train_frac, "best_val_cer": round(float(best["val_cer"]), 4),
               "best_val_wer": round(float(best["val_wer"]), 4), "best_epoch": int(best["epoch"]),
               "epochs_run": len(hist), "minutes": round(sum(float(h["secs"]) for h in hist) / 60, 2),
               "parameters": count_parameters(CRNN(charset.num_classes, mc))}
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        rows.append(row)
        el = time.time() - t_start
        if verbose:
            print(f"  [{i}/{len(todo)}] {name:28s} val CER {row['best_val_cer']:.4f}  WER {row['best_val_wer']:.4f}  "
                  f"({(time.time() - t0) / 60:.1f} min; ~{el / i * (len(todo) - i) / 60:.0f} min left)")
    return rank(rows)


def rank(rows):
    """Best first: lowest val CER, then lowest val WER, then name (deterministic)."""
    return sorted(rows, key=lambda r: (r["best_val_cer"], r["best_val_wer"], r["name"]))


def choose_settings(csv_path, defaults=DEFAULTS, useless_cer=0.99):
    """Settings for the main run: the sweep winner if there is a usable sweep, else the defaults.

    Returns (settings, info). info["source"] is "sweep" or "defaults", info["reason"] says why,
    info["winner"] is the chosen sweep row (or None). Falls back to the defaults when the CSV
    is missing/empty, or when no run learned anything (best val CER >= `useless_cer`, e.g. a
    too-short sweep), since then the ranking is noise."""
    rows = rank(load_results(csv_path)) if csv_path else []
    if not rows:
        return dict(defaults), {"source": "defaults", "winner": None, "n_runs": 0,
                                "reason": "no sweep results found, so using the default settings"}
    best = rows[0]
    if best["best_val_cer"] >= useless_cer:
        return dict(defaults), {"source": "defaults", "winner": None, "n_runs": len(rows),
                                "reason": f"no sweep run got below {useless_cer:.0%} val CER (ranking would be noise), "
                                          "so using the default settings"}
    chosen = {k: best[k] for k in SWEPT}
    return chosen, {"source": "sweep", "winner": best, "n_runs": len(rows),
                    "reason": f"best of {len(rows)} sweep runs by validation CER: {best['name']} "
                              f"(val CER {best['best_val_cer']:.4f}, WER {best['best_val_wer']:.4f} after "
                              f"{best['epochs']} epochs on {best['train_frac']:.0%} of train)"}


def make_configs(settings, **train_kw):
    """ModelConfig + TrainConfig for the chosen settings (other TrainConfig fields via train_kw)."""
    mc = ModelConfig(lstm_hidden=settings["lstm_hidden"], lstm_layers=settings["lstm_layers"],
                     dropout=settings["dropout"])
    tc = TrainConfig(lr=settings["lr"], batch_size=settings["batch_size"], **train_kw)
    return mc, tc


class RunMismatch(RuntimeError):
    pass


def check_run_dir(run_dir, model_cfg, train_cfg):
    """Refuse to resume a checkpoint that was trained with different settings.

    fit() resumes from run_dir/last.pt; if config.json there has another model or different
    lr / batch / weight decay / train fraction, resuming would silently mix two settings.
    (A different number of epochs is fine: that just trains longer.)"""
    p = Path(run_dir) / "config.json"
    if not p.exists():
        return "new run"
    old = json.loads(p.read_text())
    want_model = {"cnn_channels": list(model_cfg.cnn_channels), "lstm_hidden": model_cfg.lstm_hidden,
                  "lstm_layers": model_cfg.lstm_layers, "dropout": model_cfg.dropout}
    have_model = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in old.get("model", {}).items()}
    diffs = [f"model.{k}: checkpoint {have_model.get(k)!r} vs now {v!r}" for k, v in want_model.items()
             if have_model.get(k) != v]
    for k in ("lr", "batch_size", "weight_decay", "train_frac"):
        have = old.get("train", {}).get(k)
        now = getattr(train_cfg, k)
        if have is None or not math.isclose(float(have), float(now)):
            diffs.append(f"train.{k}: checkpoint {have} vs now {now}")
    if diffs:
        raise RunMismatch(f"{run_dir} was trained with different settings, so it can't be resumed:\n  "
                          + "\n  ".join(diffs) + "\nUse a new RUN_NAME (or delete that folder to start over).")
    return "resuming a run with the same settings" if (Path(run_dir) / "last.pt").exists() else "same settings"

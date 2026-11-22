"""CRNN vs the baselines (Tesseract, EMNIST character classifier) on exactly the same words.

Needs crnn_metrics.json + crnn_<split>_predictions.csv (Step 2) and, for each baseline that has run,
<name>_metrics.json + <name>_<split>_predictions.csv (Step 3 = tesseract, Step 4 = emnist) in the
report folder. Whichever notebook runs last builds the table with every model available.
All models are scored on the words they **all** have (matched by word ID): if a baseline was run on
a random subset of the test set, everyone is scored on that subset, so the comparison is like-for-like.
"""
import json

from src.report import errors, figures, tables
from src.report.predict import load_predictions
from src.utils.metrics import cer, wer

pct = lambda x: f"{100 * x:.2f}"

SYSTEMS = [("CRNN (ours)", "crnn"), ("Tesseract (LSTM)", "tesseract"), ("EMNIST CNN + splitter", "emnist")]
BASELINES = SYSTEMS[1:]


def _scores(rows):
    refs, hyps = [r["ref"] for r in rows], [r["hyp"] for r in rows]
    return {"cer": cer(refs, hyps), "wer": wer(refs, hyps), "words": len(rows)}


def available(d):
    """{prefix: metrics dict} for every system whose metrics + test predictions exist."""
    out = {}
    for _, p in SYSTEMS:
        if (d / f"{p}_metrics.json").exists() and (d / f"{p}_test_predictions.csv").exists():
            out[p] = json.loads((d / f"{p}_metrics.json").read_text())
    return out


def _speed(prefix, m):
    if prefix == "crnn":
        return m.get("test", {}).get("ms_per_word_batched")
    if prefix == "tesseract":
        return m.get("test", {}).get("ms_per_word_cpu_core")
    return m.get("test", {}).get("ms_per_word")


def save_comparison(rep):
    d = rep.dir
    have = available(d)
    if "crnn" not in have or len(have) < 2:
        missing = ["Step 2 (CRNN)"] if "crnn" not in have else []
        if len(have) - ("crnn" in have) < 1:
            missing.append("a baseline (Step 3 Tesseract or Step 4 EMNIST)")
        print("Comparison table not made yet; missing results from:", " and ".join(missing))
        print("It is created automatically by whichever of Steps 2-4 finishes last.")
        return None
    systems = [(n, p) for n, p in SYSTEMS if p in have]
    per_split = {}
    for split in ("val", "test"):
        preds = {p: load_predictions(d / f"{p}_{split}_predictions.csv") for _, p in systems
                 if (d / f"{p}_{split}_predictions.csv").exists()}
        if "crnn" not in preds or len(preds) < 2:
            continue
        common = set.intersection(*({r["word_id"] for r in rows} for rows in preds.values()))
        rows = {p: sorted((r for r in v if r["word_id"] in common), key=lambda r: r["word_id"]) for p, v in preds.items()}
        per_split[split] = {p: _scores(v) for p, v in rows.items()}
        per_split[split]["_rows"] = rows
    params = {"crnn": rep.numbers().get("model", {}).get("parameters"),
              "emnist": have.get("emnist", {}).get("classifier", {}).get("parameters")}
    rows, nums = [], {}
    for name, p in systems:
        row = {"Model": name}
        for split in ("val", "test"):
            s = per_split.get(split, {}).get(p)
            row[f"{split.capitalize()} CER"] = pct(s["cer"]) if s else "--"
            row[f"{split.capitalize()} WER"] = pct(s["wer"]) if s else "--"
        ms = _speed(p, have[p])
        row["ms/word"] = "" if ms is None else (f"{ms:.2f}" if ms < 10 else f"{ms:.0f}")
        row["Params"] = f"{params[p] / 1e6:.2f}M" if params.get(p) else "--"
        rows.append(row)
        nums[name] = {s: per_split[s][p] for s in per_split if p in per_split[s]}
        nums[name]["ms_per_word"] = ms
    n_test = per_split["test"]["crnn"]["words"]
    full = have["crnn"].get("test", {}).get("words")
    note = None
    if full and n_test < full:
        note = (f"All models scored on the same {n_test:,} test words (the words every model was run on); "
                f"CRNN on all {full:,} test words: CER {pct(have['crnn']['test']['cer'])}%, "
                f"WER {pct(have['crnn']['test']['wer'])}%.")
    speed = {"crnn": f"CRNN batched (128) on {have['crnn'].get('device')}",
             "tesseract": "Tesseract per word on one CPU core",
             "emnist": f"EMNIST baseline: segmentation on the CPU + batched classification on "
                       f"{have.get('emnist', {}).get('device', 'GPU')}"}
    pre = {"tesseract": "Tesseract is pretrained on printed text", "emnist": "the EMNIST CNN is trained only on EMNIST characters"}
    base_txt = "; ".join(pre[p] for _, p in systems if p in pre)
    cap = (f"CRNN vs. the baselines on the same IAM words ({n_test:,} test words from writers unseen in training). "
           f"CER and WER in %, case-sensitive, lower is better. Speed: " + "; ".join(speed[p] for _, p in systems) +
           f". Neither baseline is trained on IAM ({base_txt}).")
    paths = tables.write_table(rows, list(rows[0]), d, "table_comparison", cap, "tab:comparison", note=note)
    rep.add(paths, "Main quantitative result: CRNN vs the baselines (" + ", ".join(n for n, p in systems if p != "crnn")
            + ") CER/WER on the same val/test words, speed and size.")
    sys_rows = per_split["test"]["_rows"]
    lens = {n: errors.cer_by_length([r["ref"] for r in sys_rows[p]], [r["hyp"] for r in sys_rows[p]])
            for n, p in systems if p in sys_rows}
    p2 = figures.plot_cer_by_length(lens, d, "fig_cer_by_length_comparison",
                                    title="CER by word length: " + " vs ".join(n.split(" (")[0] for n in lens)
                                    + " (same test words)")
    rep.add(p2, "CER by word length for " + ", ".join(lens) + " on the same test words.")
    nums["note"] = note
    nums["models"] = [n for n, _ in systems]
    nums["test_words"] = n_test
    rep.numbers("comparison", nums)
    return rows

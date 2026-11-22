"""Step 2: turn a finished (or resumed) CRNN run into report figures, tables and numbers.

Everything is rebuilt from files on Drive (history.csv, best.pt) plus the evaluation, so it
works the same whether training ran in one go or was resumed after a disconnect.
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

from src.report import errors, figures, tables
from src.report.model_summary import layer_table, macs_per_word
from src.report.predict import original_image
from src.utils.metrics import cer, wer

pct = lambda x: f"{100 * x:.2f}"


def save_curves(rep, history_csv):
    hist = figures.read_history(history_csv)
    paths = figures.plot_training_curves(history_csv, rep.dir)
    rep.add(paths[0:2], "Training (blue) and validation (orange) CTC loss, and validation CER/WER per epoch; "
                        "the dotted line marks the best epoch (lowest validation CER), whose weights are kept.")
    rep.add(paths[2:4], "Training and validation CTC loss per epoch.")
    rep.add(paths[4:6], "Validation CER and WER per epoch (greedy CTC decoding).")
    return paths, hist


def save_metrics(rep, preds, eval_results, latency_ms, device_name):
    """Final val/test metrics. `preds` maps split -> rows from predict_dataset."""
    rows, nums = [], {"device": device_name, "ms_single_word": round(latency_ms, 3)}
    for split in ("val", "test"):
        p = preds[split]
        refs, hyps = [r["ref"] for r in p], [r["hyp"] for r in p]
        c, w = cer(refs, hyps), wer(refs, hyps)
        ms = eval_results[split]["ms_per_word"]
        nums[split] = {"cer": c, "wer": w, "words": len(p), "word_accuracy": 1 - w, "ms_per_word_batched": ms,
                       "loss": eval_results[split]["loss"]}
        rows.append({"Split": split, "Words": len(p), "CER (%)": pct(c), "WER (%)": pct(w),
                     "ms/word (batch 128)": f"{ms:.2f}", "ms (1 word)": f"{latency_ms:.1f}"})
    rep.numbers("crnn", nums)
    (rep.dir / "crnn_metrics.json").write_text(json.dumps(nums, indent=2))
    rep.add([str(rep.dir / "crnn_metrics.json")], "CRNN val/test CER, WER and speed (read by Step 3 for the comparison).")
    paths = tables.write_table(rows, list(rows[0]), rep.dir, "table_crnn_metrics",
                               f"CRNN results on the IAM validation and test splits (writer-independent, greedy decoding). "
                               f"Speed on {device_name}.", "tab:crnn_metrics")
    rep.add(paths, "Final CRNN metrics: CER and WER on val and test, batched throughput and single-word latency.")
    return nums


def save_model_summary(rep, model, charset, ck, train_cfg, model_cfg, history, device_name, height=32,
                       settings_info=None):
    rows, total = layer_table(model, height=height)
    macs = macs_per_word(model, height=height, width=128)
    rows_out = rows + ["midrule", {"layer": "Total", "details": f"{total * 4 / 1e6:.1f} MB as float32",
                                   "output_shape": "", "parameters": total}]
    paths = tables.write_table(
        [r if r == "midrule" else {"Layer": r["layer"], "Details": r["details"], "Output shape": r["output_shape"],
                                    "Parameters": r["parameters"]} for r in rows_out],
        ["Layer", "Details", "Output shape", "Parameters"], rep.dir, "table_model_summary",
        f"CRNN architecture. Output shapes are per image (C×H×W, or T×features); W is the input width and "
        f"T = W/4 time steps. {total:,} trainable parameters, {macs / 1e6:.0f} M multiply-accumulates for a 128-px-wide word.",
        "tab:model_summary", col_format="lllr")
    rep.add(paths, "Layer-by-layer CRNN summary with output shapes and parameter counts (model complexity).")

    best_epoch = int(ck["epoch"])
    train_secs = sum(float(r["secs"]) for r in history)
    hp = [("Optimizer", f"AdamW (weight decay {train_cfg.weight_decay:g})"),
          ("Learning rate", f"{train_cfg.lr:g}, halved after {train_cfg.lr_patience} epochs without val-CER gain"),
          ("Scheduler", "ReduceLROnPlateau (factor 0.5) on validation CER"),
          ("Early stopping", f"patience {train_cfg.patience} epochs"),
          ("Batch size", train_cfg.batch_size), ("Gradient clipping", f"max norm {train_cfg.grad_clip:g}"),
          ("Mixed precision", "fp16 on GPU" if train_cfg.amp else "off"),
          ("Epochs run / best epoch", f"{len(history)} / {best_epoch}"),
          ("Training time", f"{train_secs / 60:.1f} min on {device_name}"),
          ("CNN channels", " / ".join(map(str, model_cfg.cnn_channels))),
          ("BiLSTM", f"{model_cfg.lstm_layers} layers, hidden {model_cfg.lstm_hidden} per direction"),
          ("Dropout", f"{model_cfg.dropout:g}"),
          ("Input height", f"{height} px (width kept in proportion, max 512)"),
          ("Output classes", f"{charset.num_classes} ({len(charset.chars)} characters + CTC blank)"),
          ("Loss / decoding", "CTC (blank = 0) / greedy best path"),
          ("Parameters", f"{total:,}")]
    if settings_info:
        w = settings_info.get("winner")
        hp.append(("Settings chosen by", f"hyperparameter sweep: best of {settings_info['n_runs']} runs by validation CER "
                                         f"({w['name']})" if settings_info["source"] == "sweep" and w
                                         else "defaults (" + settings_info["reason"] + ")"))
    hp_rows = [{"Hyperparameter": k, "Value": v} for k, v in hp]
    cap = "Training setup and hyperparameters of the CRNN" + (
        " (learning rate, LSTM size/depth, dropout and batch size chosen by the hyperparameter sweep; see the sweep table)."
        if settings_info and settings_info.get("source") == "sweep" else ".")
    paths = tables.write_table(hp_rows, ["Hyperparameter", "Value"], rep.dir, "table_hyperparameters",
                               cap, "tab:hyperparameters", col_format="lp{3.6in}")
    rep.add(paths, "Hyperparameters and training setup (optimizer, schedule, epochs run, best epoch, model size).")
    nums = {"parameters": total, "size_mb_fp32": round(total * 4 / 1e6, 2), "macs_per_word_w128": macs,
            "num_classes": charset.num_classes, "charset_size": len(charset.chars), "input_height": height,
            "cnn_channels": list(model_cfg.cnn_channels), "lstm_hidden": model_cfg.lstm_hidden,
            "lstm_layers": model_cfg.lstm_layers, "dropout": model_cfg.dropout,
            "layers": [{k: v for k, v in r.items()} for r in rows]}
    rep.numbers("model", nums)
    rep.numbers("training", {"optimizer": "AdamW", "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay,
                             "scheduler": "ReduceLROnPlateau(factor=0.5, patience=%d) on val CER" % train_cfg.lr_patience,
                             "early_stopping_patience": train_cfg.patience, "batch_size": train_cfg.batch_size,
                             "grad_clip": train_cfg.grad_clip, "amp": train_cfg.amp, "max_epochs": train_cfg.epochs,
                             "epochs_run": len(history), "best_epoch": best_epoch,
                             "best_val_cer": float(ck["best_cer"]), "train_minutes": round(train_secs / 60, 2),
                             "device": device_name,
                             "settings_source": (settings_info or {}).get("source", "unknown"),
                             "settings_reason": (settings_info or {}).get("reason", "")})
    return rows, total, macs


def save_data_stats(rep, stats, datasets, label_source):
    f_rows = [("words.txt lines", stats.get("total_lines", 0)),
              ("removed: segmentation marked 'err'", stats.get("segmentation_err", 0)),
              ("removed: characters outside the charset", stats.get("chars_outside_charset", 0)),
              ("removed: label longer than 32", stats.get("too_long", 0)),
              ("removed: empty label", stats.get("empty_label", 0)),
              ("removed: missing or broken image", stats.get("bad_or_missing_image", 0)),
              ("removed: cut-off words.txt line", stats.get("malformed_line", 0)),
              ("kept (pass all filters)", stats.get("kept", 0)),
              ("not in any Aachen split", stats.get("not_in_any_split", 0))]
    paths = tables.write_table([{"Step": k, "Words": v} for k, v in f_rows], ["Step", "Words"], rep.dir,
                               "table_data_filtering", f"IAM word filtering (labels: {label_source}).",
                               "tab:data_filtering", col_format="lr")
    rep.add(paths, "How many IAM words each filter removed, and how many remain.")
    s_rows, split_nums = [], {}
    for split in ("train", "val", "test"):
        items = datasets[split].items
        writers = {it.writer for it in items if it.writer}
        forms = {it.form_id for it in items}
        mean_len = sum(len(it.text) for it in items) / max(len(items), 1)
        s_rows.append({"Split": split, "Words": len(items), "Writers": len(writers), "Forms": len(forms),
                       "Mean length": f"{mean_len:.2f}"})
        split_nums[split] = {"words": len(items), "writers": len(writers), "forms": len(forms), "mean_label_len": mean_len}
    shared = 0
    ws = {s: {it.writer for it in datasets[s].items if it.writer} for s in ("train", "val", "test")}
    shared = len((ws["train"] & ws["val"]) | (ws["train"] & ws["test"]) | (ws["val"] & ws["test"]))
    paths = tables.write_table(s_rows, list(s_rows[0]), rep.dir, "table_data_splits",
                               f"Aachen writer-independent splits after filtering; {shared} writers appear in more than one split.",
                               "tab:data_splits")
    rep.add(paths, "Words, writers and forms per split (writer-disjoint Aachen split) and mean label length.")
    rep.numbers("data", {"label_source": label_source, "filtering": {k: v for k, v in f_rows},
                         "splits": split_nums, "writers_shared_between_splits": shared,
                         "writer_source": stats.get("writer_source"), "split_method": stats.get("split_method")})
    return s_rows


def save_label_hist(rep, datasets):
    paths = figures.plot_label_length_hist([it.text for it in datasets["train"].items], rep.dir)
    rep.add(paths, "Distribution of label lengths in the training split; most IAM words are short.")
    return paths


def save_preprocessing_examples(rep, ds, n=4, seed=3):
    """At least one cleaned training sample (rubric): original crop vs. the CRNN's 32-px input."""
    rng = random.Random(seed)
    idx = [i for i in rng.sample(range(len(ds)), min(len(ds), 50)) if 3 <= len(ds.items[i].text) <= 10][:n]
    originals = [original_image(ds.items[i]) for i in idx]
    processed = [ds[i][0][0].numpy() for i in idx]
    texts = [ds.items[i].text for i in idx]
    paths = figures.plot_preprocessing_examples(originals, processed, texts, rep.dir)
    rep.add(paths, "Cleaned training samples: the original IAM word crop (left) and the CRNN input (right): grayscale, "
                   "resized to 32 px high keeping the aspect ratio, normalized to [-1, 1]; the label is shown on the left.")
    return paths


def save_prediction_grid(rep, ds, preds, n=12, seed=0, split="test", name="fig_crnn_predictions"):
    """A random (not cherry-picked) sample of words with true and predicted text."""
    idx = sorted(random.Random(seed).sample(range(len(preds)), min(n, len(preds))))
    by_id = {it.word_id: it for it in ds.items}
    rows = [preds[i] for i in idx]
    imgs = [original_image(by_id[r["word_id"]]) for r in rows]
    right = sum(r["ref"] == r["hyp"] for r in rows)
    paths = figures.plot_prediction_grid(imgs, [r["ref"] for r in rows], [r["hyp"] for r in rows], rep.dir, name)
    rep.add(paths, f"CRNN predictions on {len(rows)} randomly chosen {split} words from unseen writers "
                   f"({right}/{len(rows)} exactly right); green = correct, red = wrong.")
    return paths


def save_error_analysis(rep, preds, system="CRNN", prefix="crnn", k=15):
    refs, hyps = [r["ref"] for r in preds], [r["hyp"] for r in preds]
    conf = errors.char_confusions(refs, hyps)
    bd = errors.error_breakdown(refs, hyps)
    subs = Counter({pair: n for pair, n in conf.items() if errors.EMPTY not in pair})
    paths = figures.plot_char_confusions(subs, rep.dir, f"fig_{prefix}_char_confusions", k=k,
                                         title=f"{system}: top substitutions (test)\n"
                                               f"{bd['substitutions']:,} sub. / {bd['deletions']:,} del. / {bd['insertions']:,} ins.")
    rep.add(paths, f"{system}: the {k} most frequent character substitutions (true → predicted) on the test split; "
                   "the title gives the total substitutions, deletions and insertions behind its CER.")
    rows = [{"True": a, "Predicted": b, "Count": n} for (a, b), n in conf.most_common(k)]
    p2 = tables.write_table(rows, ["True", "Predicted", "Count"], rep.dir, f"table_{prefix}_char_confusions",
                            f"{system}: most frequent character errors on the test split (∅ = missing/extra character).",
                            f"tab:{prefix}_char_confusions", col_format="ccr")
    rep.add(p2, f"{system} top character confusions as a table.")
    wm = [{"True word": r, "Predicted": h if h else "(empty)", "Count": n} for (r, h), n in errors.word_mistakes(refs, hyps, k)]
    p3 = tables.write_table(wm, ["True word", "Predicted", "Count"], rep.dir, f"table_{prefix}_word_errors",
                            f"{system}: most frequent wrong words on the test split.", f"tab:{prefix}_word_errors",
                            col_format="llr")
    rep.add(p3, f"{system} most common whole-word mistakes (qualitative error analysis).")
    rep.numbers(f"{prefix}_error_breakdown", bd)
    return conf, bd, paths + p2 + p3


def save_cer_by_length(rep, preds, system="CRNN", prefix="crnn"):
    rows = errors.cer_by_length([r["ref"] for r in preds], [r["hyp"] for r in preds])
    paths = figures.plot_cer_by_length({system: rows}, rep.dir, f"fig_{prefix}_cer_by_length",
                                       title=f"{system}: CER by word length (test)")
    rep.add(paths, f"{system}: CER on the test split grouped by word length (number of words per group in brackets).")
    p2 = tables.write_table([{"Length": r["length"], "Words": r["n"], "CER (%)": pct(r["cer"]), "WER (%)": pct(r["wer"])}
                             for r in rows], ["Length", "Words", "CER (%)", "WER (%)"], rep.dir,
                            f"table_{prefix}_cer_by_length", f"{system} error rates by word length (test).",
                            f"tab:{prefix}_cer_by_length")
    rep.add(p2, f"{system} CER/WER per word-length group (numbers behind the bar chart).")
    rep.numbers(f"{prefix}_cer_by_length", rows)
    return paths + p2


def save_architecture(rep, model_cfg, num_classes):
    paths = figures.draw_crnn_diagram(model_cfg, num_classes, rep.dir)
    rep.add(paths, "CRNN architecture: five convolutional blocks reduce the 32-px-high image to one feature column "
                   "per 4 pixels of width, a 2-layer BiLSTM reads the sequence, and a linear layer outputs "
                   f"{num_classes} classes per time step for CTC.")
    return paths


def save_sweep(rep, sweep_csv, chosen_name=None):
    """table_sweep (all swept settings, best first, chosen row in bold) + fig_sweep."""
    from src.training.sweep import load_results, rank
    rows = rank(load_results(sweep_csv))
    if not rows:
        print("no sweep results (RUN_SWEEP was False and no sweep_results.csv): skipping the sweep table")
        return None
    out = []
    for i, r in enumerate(rows, 1):
        chosen = r["name"] == chosen_name
        out.append({"#": f"{i}*" if chosen else i, "LR": f"{r['lr']:g}", "Hidden": r["lstm_hidden"],
                    "Layers": r["lstm_layers"], "Dropout": f"{r['dropout']:g}", "Batch": r["batch_size"],
                    "Val CER (%)": pct(r["best_val_cer"]), "Val WER (%)": pct(r["best_val_wer"]),
                    "Best ep.": r["best_epoch"], "Min": f"{r['minutes']:.1f}",
                    "Chosen": "yes" if chosen else "", "_bold": chosen})
    ep, fr = rows[0]["epochs"], rows[0]["train_frac"]
    cols = ["#", "LR", "Hidden", "Layers", "Dropout", "Batch", "Val CER (%)", "Val WER (%)", "Best ep.", "Min"]
    note = "* = setting used for the final model." if chosen_name else "No run was chosen (defaults used)."
    paths = tables.write_table(out, cols + ["Chosen"], rep.dir, "table_sweep", "", "tab:sweep")
    # the .tex leaves out the Chosen column (the bold row and * mark it); the CSV keeps it
    short = {"Dropout": "Drop.", "Val CER (%)": "CER (%)", "Val WER (%)": "WER (%)", "Best ep.": "Ep."}
    tex_rows = [{short.get(k, k): v for k, v in r.items()} for r in out]
    tex = tables.to_latex(tex_rows, [short.get(c, c) for c in cols],
                          f"Hyperparameter sweep, ranked by validation CER (test set never used). Each run: {ep} epochs on "
                          f"{fr:.0%} of the training words; CER/WER on the full validation set; Ep. = best epoch; Min = minutes. A small grid over learning rate and LSTM "
                          "hidden size, plus one-at-a-time changes of dropout, batch size, hidden size and LSTM depth "
                          "around the default (lr 0.001, hidden 256, dropout 0.3, batch 64, 2 layers).",
                          "tab:sweep", note=note)
    Path(paths[1]).write_text(tex)
    rep.add(paths, f"Hyperparameter sweep ({len(rows)} runs, {ep} epochs on {fr:.0%} of train each), ranked by "
                   "validation CER; the bold row (*) is the setting used for the final model.")
    p2 = figures.plot_sweep(rows, rep.dir, chosen=chosen_name)
    rep.add(p2, "Best validation CER of every sweep run; the orange bar is the setting chosen for the final model.")
    rep.numbers("sweep", {"runs": rows, "chosen": chosen_name, "epochs_per_run": ep, "train_frac": fr,
                          "total_minutes": round(sum(r["minutes"] for r in rows), 1)})
    return paths + p2

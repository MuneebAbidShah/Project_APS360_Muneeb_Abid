"""Step 4: report figures, tables and numbers for the EMNIST character-classifier baseline."""
import json
import random

from src.baselines import char_segment as cs
from src.baselines.emnist import CLASSES
from src.report import figures, tables
from src.report.predict import original_image

pct = lambda x: f"{100 * x:.2f}"


def save_classifier(rep, info):
    """EMNIST classifier on its own: test accuracy (overall / per group / case-insensitive), size, training."""
    t = info["test"]
    rows = [{"Item": "Training data", "Value": f"EMNIST byclass ({len(CLASSES)} classes: 0-9, A-Z, a-z), "
                                               f"{info['cfg']['train_images']:,} training images"},
            {"Item": "Test accuracy", "Value": f"{pct(t['accuracy'])}% ({t['n']:,} EMNIST test images)"},
            {"Item": "Test accuracy, case-insensitive", "Value": f"{pct(t['accuracy_case_insensitive'])}%"},
            {"Item": "Digits / upper / lower", "Value": " / ".join(f"{pct(t['accuracy_' + g])}%" for g in ("digits", "upper", "lower"))},
            {"Item": "Parameters", "Value": f"{info['parameters']:,}"},
            {"Item": "Training", "Value": f"{len(info['history'])} epochs (best {info['best_epoch']}), "
                                          f"{info['minutes']:.1f} min on {info['device']}"}]
    paths = tables.write_table(rows, ["Item", "Value"], rep.dir, "table_emnist_classifier",
                               "The EMNIST character classifier on its own (EMNIST byclass test set). Upper/lower-case "
                               "pairs such as o/O, s/S and c/C look almost the same, which is why byclass accuracy "
                               "is capped near 88% and the case-insensitive accuracy is higher.",
                               "tab:emnist_classifier", col_format="lp{3.45in}")
    rep.add(paths, "EMNIST character classifier: test accuracy overall, case-insensitive and per group, size and training time.")
    return paths


def save_metrics(rep, preds, infos, full_sizes, tuning, settings, device, classifier):
    """Word-level CER/WER on IAM val/test (same words and code as the CRNN), plus the val tuning table."""
    from src.baselines.emnist_pipeline import scores
    nums = {"settings": settings, "device": device, "classifier": {k: v for k, v in classifier.items() if k != "history"}}
    rows = []
    for split, p in preds.items():
        s = scores(p)
        alnum = [r for r in p if r["ref"].isalnum()]
        nums[split] = {**s, "of_words": full_sizes[split], "skipped": infos[split]["skipped"],
                       "ms_per_word": infos[split]["ms_per_word"], "seg_ms_per_word": infos[split]["seg_ms_per_word"],
                       "empty_outputs": sum(1 for r in p if not r["hyp"]),
                       "cer_letters_digits_only_words": scores(alnum)["cer"] if alnum else None}
        rows.append({"Split": split, "Words": f"{len(p):,} of {full_sizes[split]:,}", "CER (%)": pct(s["cer"]),
                     "WER (%)": pct(s["wer"]), "CER, no case (%)": pct(s["cer_case_insensitive"]),
                     "ms/word": f"{infos[split]['ms_per_word']:.1f}"})
    rep.numbers("emnist", nums)
    (rep.dir / "emnist_metrics.json").write_text(json.dumps(nums, indent=2))
    rep.add([str(rep.dir / "emnist_metrics.json")], "EMNIST baseline: val/test CER, WER (also case-insensitive), speed, settings.")
    paths = tables.write_table(rows, list(rows[0]), rep.dir, "table_emnist_metrics",
                               f"EMNIST character-classifier baseline (rule-based letter splitter + CNN trained only on "
                               f"EMNIST) on the IAM validation and test words. CER/WER are case-sensitive, as for the other "
                               f"models; 'no case' ignores upper/lower case. ms/word = segmentation on the CPU + "
                               f"classification on {device}.", "tab:emnist_metrics")
    rep.add(paths, "Baseline metrics: EMNIST baseline CER/WER on val and test (case-sensitive like the other models, "
                   "plus case-insensitive CER) and time per word.")
    trows = [{"Width factor": f"{r['width_factor']:.1f}", "IAM prior": "on" if r["prior"] else "off",
              "Val CER (%)": pct(r["cer"]), "Val WER (%)": pct(r["wer"]), "_bold": i == 0} for i, r in enumerate(tuning)]
    p2 = tables.write_table(trows, ["Width factor", "IAM prior", "Val CER (%)", "Val WER (%)"], rep.dir,
                            "table_emnist_tuning",
                            "EMNIST baseline settings chosen on validation words (best first, chosen in bold): the "
                            "expected character width as a multiple of the core (x-)height, and whether the classifier's "
                            "scores are re-weighted by IAM training-set character frequencies.", "tab:emnist_tuning")
    rep.add(p2, "EMNIST baseline tuning on validation words: splitter width factor and the IAM character prior (chosen row in bold).")
    return nums


def segmentation_examples(ds, segs, logps, prior, alpha, n=4, seed=0):
    """Pick `n` random test words (not cherry-picked) and collect what the figure needs."""
    idx = random.Random(seed).sample(range(len(segs)), min(n, len(segs)))
    out = []
    for i in idx:
        s, lp = segs[i], logps[i]
        it = s["item"]
        hyp = cs.decode(s["pieces"], lp, prior, alpha)
        sc = lp if (prior is None or not alpha) else lp + alpha * prior[None, :]
        letters = iter(CLASSES[int(k)] for k in (sc.argmax(1) if len(sc) else []))
        preds = [q["char"] if q["kind"] == "punct" else next(letters) for q in s["pieces"]]
        out.append({"gray": cs.deslanted_gray(original_image(it), s.get("shear", 0.0)), "pieces": s["pieces"], "crops": s["crops"], "preds": preds,
                    "ref": it.text, "hyp": hyp})
    return out


def save_segmentation_figure(rep, examples):
    paths = figures.plot_segmentation_examples(examples, rep.dir)
    rep.add(paths, f"EMNIST baseline on {len(examples)} randomly chosen test words (shown after slant correction): the splitter's character boxes "
                   "(red = classified crop, blue = punctuation rule), the normalized 28x28 crops and the predicted characters.")
    return paths


def save_failure_grid(rep, ds, preds, n=12, seed=0):
    wrong = [r for r in preds if r["ref"] != r["hyp"]]
    rows = random.Random(seed).sample(wrong, min(n, len(wrong)))
    if not rows:
        print("EMNIST baseline got every word right: no failure grid"); return None
    by_id = {it.word_id: it for it in ds.items}
    imgs = [original_image(by_id[r["word_id"]]) for r in rows]
    paths = figures.plot_prediction_grid(imgs, [r["ref"] for r in rows], [r["hyp"] for r in rows], rep.dir,
                                         "fig_emnist_failures")
    rep.add(paths, f"Typical EMNIST-baseline failures: {len(rows)} randomly chosen wrong test predictions (red).")
    return paths


def save_pipeline(rep):
    paths = figures.draw_emnist_pipeline(rep.dir)
    rep.add(paths, "EMNIST baseline pipeline: binarize, connected components, rule-based splitting/merging, "
                   "EMNIST-style normalization, a CNN trained only on EMNIST, then the characters are joined "
                   "and scored with the same CER/WER code as the CRNN.")
    return paths

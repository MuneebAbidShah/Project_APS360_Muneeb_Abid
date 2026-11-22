"""Step 3: report figures, tables and numbers for the Tesseract baseline."""
import json
import random

from src.baselines import tesseract as tess
from src.report import figures, tables
from src.report.predict import original_image
from src.utils.metrics import cer, wer

pct = lambda x: f"{100 * x:.2f}"


def save_metrics(rep, preds, secs, workers, full_sizes, device="CPU"):
    """`preds`: split -> rows (word_id, ref, hyp); `secs`: split -> wall-clock seconds."""
    rows, nums = [], {"version": tess.version(), "config": tess.config(), "border_px": tess.BORDER,
                      "workers": workers, "device": device}
    for split, p in preds.items():
        refs, hyps = [r["ref"] for r in p], [r["hyp"] for r in p]
        c, w = cer(refs, hyps), wer(refs, hyps)
        ms_wall = 1000 * secs[split] / max(len(p), 1)   # wall-clock per word with `workers` in parallel
        nums[split] = {"cer": c, "wer": w, "words": len(p), "of_words": full_sizes[split],
                       "empty_outputs": sum(1 for h in hyps if not h), "ms_per_word_wall": ms_wall,
                       "ms_per_word_cpu_core": ms_wall * workers}
        rows.append({"Split": split, "Words": f"{len(p):,} of {full_sizes[split]:,}", "CER (%)": pct(c), "WER (%)": pct(w),
                     "Empty": nums[split]["empty_outputs"], "ms/word": f"{ms_wall * workers:.0f}"})
    nums["ms_per_word_wall"] = nums.get("test", nums[next(iter(preds))])["ms_per_word_wall"]
    rep.numbers("tesseract", nums)
    (rep.dir / "tesseract_metrics.json").write_text(json.dumps(nums, indent=2))
    rep.add([str(rep.dir / "tesseract_metrics.json")], "Tesseract val/test CER, WER and speed.")
    paths = tables.write_table(rows, list(rows[0]), rep.dir, "table_tesseract_metrics",
                               f"Tesseract {nums['version']} baseline (LSTM engine, {tess.config()}, "
                               f"{tess.BORDER}-px white border) on the IAM validation and test splits (random subsets if fewer words "
                               f"than the split). Empty = no text returned; ms/word = time per word on one CPU core.",
                               "tab:tesseract_metrics")
    rep.add(paths, "Baseline metrics: Tesseract CER/WER on val and test, empty outputs, speed per word per CPU core.")
    return nums


def save_failure_grid(rep, ds, preds, n=12, seed=0):
    wrong = [r for r in preds if r["ref"] != r["hyp"]]
    rows = random.Random(seed).sample(wrong, min(n, len(wrong)))
    if not rows:
        print("Tesseract got every word right: no failure grid"); return None
    by_id = {it.word_id: it for it in ds.items}
    imgs = [original_image(by_id[r["word_id"]]) for r in rows]
    paths = figures.plot_prediction_grid(imgs, [r["ref"] for r in rows], [r["hyp"] for r in rows], rep.dir,
                                         "fig_tesseract_failures")
    rep.add(paths, f"Typical Tesseract failures: {len(rows)} randomly chosen wrong test predictions "
                   "(red = predicted text; '(empty)' = Tesseract returned nothing).")
    return paths


def save_pipeline(rep):
    paths = figures.draw_tesseract_pipeline(rep.dir, border=tess.BORDER, psm=tess.PSM, oem=tess.OEM)
    rep.add(paths, "Tesseract baseline pipeline: light preprocessing (white border), the pretrained LSTM engine "
                   "in single-word mode, output normalization, and scoring with the same CER/WER code as the CRNN.")
    return paths

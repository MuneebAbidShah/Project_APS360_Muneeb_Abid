"""EMNIST baseline end to end on IAM words: segment -> normalize -> classify -> join -> score.

Settings that are tuned (character width factor, and whether to use the IAM character prior) are
chosen on **validation** words only; the test split is run once with the chosen settings.
Unreadable images are skipped and listed, exactly like the CRNN and Tesseract runs.
"""
import time
from dataclasses import replace

import numpy as np
import torch

from src.baselines import char_segment as cs
from src.baselines.emnist import CLASSES, predict_logits
from src.utils.metrics import cer, wer


def segment_items(items, params=cs.SegParams(), progress=True, quiet=False):
    """Segment every word. Returns (segs, skipped_ids, seconds); segs keep pieces (without masks)
    and the 28x28 crops."""
    segs, skipped, t0 = [], [], time.time()
    step = max(1, len(items) // 5)
    for i, it in enumerate(items):
        g = cs.load_gray(it.path)
        if g is None:
            skipped.append(it.word_id); continue
        pieces, crops, info = cs.segment_word(g, params, return_info=True)
        segs.append({"item": it, "pieces": [{k: v for k, v in q.items() if k != "mask"} for q in pieces],
                     "crops": crops, "shear": info.get("shear", 0.0)})
        if progress and (i + 1) % step == 0:
            el = time.time() - t0
            print(f"  segmented {i + 1:,}/{len(items):,} words, {el:.0f}s elapsed, ~{el / (i + 1) * (len(items) - i - 1):.0f}s left")
    if skipped and not quiet:
        print(f"  skipped {len(skipped)} unreadable image(s): {', '.join(skipped[:10])}")
    return segs, skipped, time.time() - t0


def classify(segs, model, device):
    """log-probabilities for every crop, split back per word. Returns (list of arrays, seconds)."""
    t0 = time.time()
    allc = [s["crops"] for s in segs]
    sizes = [len(c) for c in allc]
    flat = np.concatenate(allc) if sum(sizes) else np.zeros((0, 28, 28), np.uint8)
    lp = torch.log_softmax(predict_logits(model, flat, device), 1).numpy() if len(flat) else \
        np.zeros((0, len(CLASSES)), np.float32)
    out, k = [], 0
    for n in sizes:
        out.append(lp[k:k + n]); k += n
    return out, time.time() - t0


def decode_all(segs, logps, prior=None, alpha=0.0):
    rows = []
    for s, lp in zip(segs, logps):
        it = s["item"]
        h = cs.decode(s["pieces"], lp, prior, alpha)
        rows.append({"word_id": it.word_id, "writer": it.writer or "", "ref": it.text, "hyp": h,
                     "correct": str(int(h == it.text)), "pieces": len(s["pieces"])})
    return rows


def scores(rows):
    refs, hyps = [r["ref"] for r in rows], [r["hyp"] for r in rows]
    return {"cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "cer_case_insensitive": cer([r.lower() for r in refs], [h.lower() for h in hyps]),
            "wer_case_insensitive": wer([r.lower() for r in refs], [h.lower() for h in hyps]),
            "words": len(rows)}


def tune(val_items, model, device, prior, width_factors=(6.0, 7.0, 8.0, 9.0, 10.0, 11.0), alphas=(0.0, 1.0),
         base=cs.SegParams()):
    """Grid over the width factor and prior on/off, scored by validation CER. Returns
    (best_params, best_alpha, table rows sorted best first)."""
    table = []
    for wf in width_factors:
        p = replace(base, width_factor=wf)
        segs, _, _ = segment_items(val_items, p, progress=False, quiet=True)
        logps, _ = classify(segs, model, device)
        for a in alphas:
            s = scores(decode_all(segs, logps, prior, a))
            table.append({"width_factor": wf, "prior": a, **s})
            print(f"  width factor {wf:.1f} x stroke | prior {'on ' if a else 'off'} | val CER {s['cer']:.4f} | WER {s['wer']:.4f}")
    table.sort(key=lambda r: (r["cer"], r["wer"]))
    best = table[0]
    return replace(base, width_factor=best["width_factor"]), best["prior"], table


def run(items, model, device, params, prior=None, alpha=0.0, progress=True):
    """Recognize `items`. Returns (rows, info) with timing (segmentation on CPU, classification
    batched on `device`) and the skipped word IDs."""
    segs, skipped, t_seg = segment_items(items, params, progress)
    logps, t_cls = classify(segs, model, device)
    rows = decode_all(segs, logps, prior, alpha)
    n = max(len(rows), 1)
    info = {"skipped": skipped, "seg_ms_per_word": 1000 * t_seg / n, "cls_ms_per_word": 1000 * t_cls / n,
            "ms_per_word": 1000 * (t_seg + t_cls) / n, "crops": int(sum(len(s["crops"]) for s in segs)),
            "segs": segs, "logps": logps}
    return rows, info


FIELDS = ["word_id", "writer", "ref", "hyp", "correct", "pieces"]


def save_rows(rows, path):
    import csv
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)
    return str(path)

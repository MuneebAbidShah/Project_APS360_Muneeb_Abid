"""Predictions with word IDs (so CRNN and Tesseract can be compared on exactly the same words)."""
import csv

import numpy as np
import torch
from PIL import Image

from src.data.iam import collate_fn
from src.models.crnn import CRNN
from src.training.sampler import BucketBatchSampler


@torch.no_grad()
def predict_dataset(model, ds, charset, device, batch_size=128, amp=False):
    """Greedy-decode every word of `ds`, batched exactly like evaluate() (same buckets, same
    padding, so the same predictions). Returns rows in dataset order."""
    model.eval()
    out = {}
    for idx in BucketBatchSampler(ds.widths(), batch_size, shuffle=False):
        samples = [(i, ds[i]) for i in idx]
        idx = [i for i, smp in samples if smp is not None]     # unreadable images are skipped
        if not idx:
            continue
        batch = collate_fn([smp for _, smp in samples if smp is not None])
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            lp = model(batch["images"].to(device))
        best = lp.float().argmax(2).cpu()
        L = CRNN.output_lengths(batch["widths"])
        for b, i in enumerate(idx):
            it = ds.items[i]
            out[i] = {"word_id": it.word_id, "writer": it.writer or "", "ref": it.text,
                      "hyp": charset.ctc_greedy_decode(best[: L[b], b].tolist())}
    return [out[i] for i in sorted(out)]


def save_predictions(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["word_id", "writer", "ref", "hyp", "correct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "correct": int(r["ref"] == r["hyp"])})
    return str(path)


def load_predictions(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def original_image(item):
    """The word image at its original resolution (uint8, white background), for figures."""
    try:
        with Image.open(item.path) as im:
            return np.asarray(im.convert("L"))
    except Exception:
        return np.full((32, 64), 255, dtype=np.uint8)   # unreadable file: blank placeholder (figures only)

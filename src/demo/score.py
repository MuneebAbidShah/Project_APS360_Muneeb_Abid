"""Score the webcam test set (captured with live_demo.py) with every model.

All models read the SAME cleaned image of each word (webcam.clean_photo, as in the live demo):
  - CRNN: the training transform + greedy CTC, exactly as in the demo;
  - Tesseract: the same settings as Step 3 (white 16-px border, --oem 1 --psm 8);
  - EMNIST baseline: the same splitter settings and IAM prior that Step 4 chose on validation.
Rows marked needs_check=1 (PAGE-mode page captures, labelled with the model's guess) are skipped
until the user has corrected the label and set needs_check to 0.
Two CRNNs are scored: Model A (live_demo/crnn_model.pt, trained on clean IAM) and Model B
(outputs/model_b/crnn_model_b.pt, same architecture and settings, trained with camera-style augmentation);
architecture_diff / config_diff check that Model B really has Model A's architecture before scoring.
An extra CRNN row without the photo clean-up (raw guide-box crop) shows what the clean-up is worth.
Labels are compared case-sensitively with the same CER/WER code as everywhere else.
"""
import csv
import os
import time

import numpy as np

from src.demo.webcam import clean_photo, to_gray
from src.utils.metrics import cer, wer


def find_webcam_dirs(project_dir, max_depth=3):
    """Folders called webcam_test* (with a labels.csv) under the project folder, most rows first."""
    hits = []
    base = os.path.abspath(project_dir)
    for root, dirs, files in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        if os.path.basename(root).startswith("webcam_test") and "labels.csv" in files:
            hits.append((len(read_labels(root)), root))
    return [r for _, r in sorted(hits, key=lambda t: -t[0])]


def read_labels(folder, unchecked=None):
    """[{'file', 'label', ...}] for rows whose image exists (paths in labels.csv are relative).
    Rows with needs_check=1 (PAGE-mode captures whose label is only the model's guess) are left out
    and their files appended to `unchecked`: a guess is not a true label."""
    p = os.path.join(folder, "labels.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("label") or "").strip() and r.get("file")]
    out = []
    for r in rows:
        if str(r.get("needs_check") or "0").strip() not in ("", "0"):
            if unchecked is not None:
                unchecked.append(r["file"])
            continue
        path = os.path.join(folder, r["file"].replace("\\", "/"))
        if os.path.exists(path):
            out.append({**r, "label": r["label"].strip(), "path": path})
    return out


def load_rows(folder, unchecked=None):
    import cv2
    rows, unreadable = [], []
    for r in read_labels(folder, unchecked):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None or not img.size:
            unreadable.append(r["file"]); continue
        rows.append({**r, "raw": img, "clean": clean_photo(img)})
    return rows, unreadable


def iam_train_texts(words_txt, split_dir, charset_chars, max_len=32):
    """IAM training labels (Aachen train forms, status ok), only to rebuild Step 4's character prior."""
    from src.data.iam import parse_words_txt, word_id_parts
    train = {l.strip() for l in open(os.path.join(split_dir, "train.uttlist")) if l.strip()}
    ok = set(charset_chars)
    return [t for wid, st, t in parse_words_txt(words_txt) if st == "ok" and len(t) <= max_len
            and set(t) <= ok and word_id_parts(wid)[0] in train]


MODEL_A, MODEL_B = "crnn", "crnn_b"     # keys used in the hyps / metrics dicts


def run_crnn(rows, rec, with_raw=True):
    """-> (predictions on the cleaned crops, predictions on the raw crops or None, ms per word)."""
    t = time.time()
    hyp = [rec.predict_tensor(rec.tensor(r["clean"])) if r["clean"] is not None else "" for r in rows]
    raw = [rec.predict_tensor(rec.tensor(to_gray(r["raw"]))) for r in rows] if with_raw else None
    return hyp, raw, 1000 * (time.time() - t) / max(1, (2 if with_raw else 1) * len(rows))


def _norm_cfg(cfg):
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(cfg).items()}


def architecture_diff(rec_a, rec_b):
    """Differences between two loaded Recognizers that would make the comparison unfair or the
    weights incompatible: model settings, charset, preprocessing. [] = same architecture."""
    out = []
    a, b = _norm_cfg(rec_a.model_cfg), _norm_cfg(rec_b.model_cfg)
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"model_cfg.{k}: A={a.get(k)!r} B={b.get(k)!r}")
    if list(rec_a.charset.chars) != list(rec_b.charset.chars):
        out.append("charset differs")
    pa, pb = dict(rec_a.preprocess_cfg), dict(rec_b.preprocess_cfg)
    for k in sorted(set(pa) | set(pb)):
        if pa.get(k) != pb.get(k):
            out.append(f"preprocess.{k}: A={pa.get(k)!r} B={pb.get(k)!r}")
    return out


def config_diff(rec, config):
    """Compare a loaded model with a training run's config.json ({'model': {...}} or the model dict itself):
    every key the config gives must match the model settings stored in the checkpoint."""
    want = _norm_cfg(config.get("model", config))
    have = _norm_cfg(rec.model_cfg)
    return [f"{k}: checkpoint={have.get(k)!r} config.json={v!r}" for k, v in want.items() if have.get(k) != v]


def run_tesseract(rows):
    import pytesseract
    from PIL import Image
    from src.baselines import tesseract as T
    t, out = time.time(), []
    for r in rows:
        if r["clean"] is None:
            out.append(""); continue
        img = T.preprocess(Image.fromarray(r["clean"]))
        out.append(T.normalize(pytesseract.image_to_string(img, lang="eng", config=T.config())))
    return out, 1000 * (time.time() - t) / max(1, len(rows))


def run_emnist(rows, clf, device, params, prior=None, alpha=0.0):
    from src.baselines import char_segment as cs
    from src.baselines.emnist_pipeline import classify
    t, segs, idx = time.time(), [], []
    for i, r in enumerate(rows):
        if r["clean"] is not None:
            pieces, crops = cs.segment_word(r["clean"], params)
            segs.append({"pieces": pieces, "crops": crops}); idx.append(i)
    logps, _ = classify(segs, clf, device) if segs else ([], 0)
    out = [""] * len(rows)
    for i, s, lp in zip(idx, segs, logps):
        out[i] = cs.decode(s["pieces"], lp, prior, alpha)
    return out, 1000 * (time.time() - t) / max(1, len(rows))


def scores(refs, hyps):
    return {"cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "cer_case_insensitive": cer([r.lower() for r in refs], [h.lower() for h in hyps]),
            "word_accuracy": float(np.mean([r == h for r, h in zip(refs, hyps)])) if refs else 0.0,
            "words": len(refs)}

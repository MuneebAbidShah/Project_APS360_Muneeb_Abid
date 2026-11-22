"""Step 5: report outputs for the webcam test set (words written on paper and captured by a phone
camera with live_demo.py), scored with the same CER/WER code as the IAM results."""
import csv
import json
import random

import numpy as np

from src.report import figures, tables
from src.report.compare import SYSTEMS
from src.report.style import FULL_WIDTH, save_figure

pct = lambda x: f"{100 * x:.2f}"
NAMES = dict((p, n) for n, p in SYSTEMS)            # names used in report_numbers.json["comparison"] (IAM results)
RAW = "crnn_no_cleanup"
MODEL_B = "crnn_b"
ORDER = ["tesseract", "emnist", "crnn", MODEL_B]     # rows of the main webcam table (baselines first)
DISPLAY = {"tesseract": "Tesseract (LSTM)", "emnist": "EMNIST CNN + splitter",
           "crnn": "CRNN, Model A", MODEL_B: "CRNN + augmentation, Model B",
           RAW: "CRNN Model A, raw crop (no clean-up)"}


def save_predictions(rep, rows, hyps):
    """webcam_predictions.csv (one column per model) + webcam_predictions_<model>.csv (one file per model:
    file, label, prediction, exact match, CER of that word)."""
    from src.utils.metrics import cer
    path = rep.dir / "webcam_predictions.csv"
    cols = ["file", "label"] + list(hyps)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(rows):
            w.writerow([r["file"], r["label"]] + [hyps[k][i] for k in hyps])
    rep.add([str(path)], "Webcam test set: every captured word with its true label and each model's prediction "
                         f"({', '.join(hyps)}; '{RAW}' = CRNN on the raw guide-box crop, without the photo clean-up).")
    per_model = []
    for k, hy in hyps.items():
        pm = rep.dir / f"webcam_predictions_{k}.csv"
        with open(pm, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "prediction", "correct", "cer"])
            for r, h in zip(rows, hy):
                w.writerow([r["file"], r["label"], h, int(r["label"] == h), f"{cer([r['label']], [h]):.4f}"])
        per_model.append(str(pm))
    rep.add(per_model, "Webcam test set, one file per model: true label, prediction, exact match (1/0) and the "
                       "word's CER.")
    return str(path)


def save_metrics(rep, rows, hyps, ms, info, iam_cer=None):
    """hyps: {'crnn': [...], 'crnn_b': [...], 'crnn_no_cleanup': [...], 'tesseract': [...], 'emnist': [...]}
    (same order as rows; any may be missing). iam_cer: {key: IAM test CER} for models that are not in
    report_numbers.json["comparison"] (Model B, from outputs/model_b/model_b_metrics.json).
    Writes table_webcam_metrics (the 4 models: Tesseract, EMNIST, Model A, Model B) and, if the raw-crop
    row exists, table_webcam_cleanup_ablation (Model A with vs without the photo clean-up)."""
    from src.demo.score import scores
    refs = [r["label"] for r in rows]
    iam = rep.numbers().get("comparison", {})
    iam_cer = dict(iam_cer or {})
    nums = {"words": len(rows), **info, "models": {}}
    main_rows, abl_rows = [], []
    keys = [k for k in ORDER if k in hyps] + ([RAW] if RAW in hyps else [])
    for key in keys:
        s = scores(refs, hyps[key])
        name = DISPLAY[key]
        ic = iam_cer.get(key)
        if ic is None and key in NAMES:
            ic = iam.get(NAMES[key], {}).get("test", {}).get("cer")
        nums["models"][key] = {**s, "name": name, "ms_per_word": ms.get(key), "iam_test_cer": ic,
                               "empty_outputs": sum(1 for h in hyps[key] if not h)}
        row = {"Model": name, "CER": pct(s["cer"]), "WER": pct(s["wer"]),
               "CER no case": pct(s["cer_case_insensitive"]), "Word acc.": pct(s["word_accuracy"]),
               "IAM test CER": pct(ic) if ic is not None else "--"}
        (abl_rows if key == RAW else main_rows).append(row)
        if key == "crnn":
            abl_rows.insert(0, {**row, "Model": "CRNN Model A, cleaned crop (as in the demo)"})
    rep.numbers("webcam", nums)
    (rep.dir / "webcam_metrics.json").write_text(json.dumps(nums, indent=2))
    rep.add([str(rep.dir / "webcam_metrics.json")], "Webcam test set: CER/WER of every model, number of words, settings.")
    cols = ["Model", "CER", "WER", "CER no case", "Word acc.", "IAM test CER"]
    cap = (f"Webcam test set: {len(rows)} handwritten words photographed with a phone camera (DroidCam) and "
           f"captured with the live demo; every model reads the same cleaned-up crop. All numbers in %; CER/WER are "
           f"case-sensitive, lower is better; 'no case' ignores upper/lower case; word acc. = exactly right words. "
           f"IAM test CER (same models, IAM test words) for comparison. Model B = Model A's architecture and "
           f"settings trained with camera-style augmentation.")
    paths = tables.write_table(main_rows, cols, rep.dir, "table_webcam_metrics", cap, "tab:webcam")
    rep.add(paths, "Webcam test set results: Tesseract, EMNIST baseline, CRNN Model A and Model B (augmented) on the "
                   "same captured words, next to their IAM test CER (domain gap).")
    if RAW in hyps and len(abl_rows) == 2:
        p2 = tables.write_table(abl_rows, cols, rep.dir, "table_webcam_cleanup_ablation",
                                "Webcam test set: CRNN Model A on the cleaned crop (as in the live demo) vs on the raw "
                                "guide-box crop (no photo clean-up). All numbers in %.", "tab:webcam_cleanup")
        rep.add(p2, "Photo clean-up ablation on the webcam set (Model A with vs without the clean-up).")
    return nums


def save_vs_iam_figure(rep, nums):
    ms = [(v["name"], v["iam_test_cer"], v["cer"]) for k, v in nums["models"].items() if k != RAW]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.7, 2.0))
    x = np.arange(len(ms))
    a = ax.bar(x - 0.18, [100 * (m[1] or 0) for m in ms], 0.36, label="IAM test", color=figures.BLUE)
    b = ax.bar(x + 0.18, [100 * m[2] for m in ms], 0.36, label=f"Webcam ({nums['words']} words)", color=figures.ORANGE)
    for bars in (a, b):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()), ha="center",
                        va="bottom", fontsize=6.5, xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels([m[0].replace(" (", "\n(").replace(" + ", "\n+ ") for m in ms], fontsize=7)
    ax.set_ylabel("CER (%)"); ax.legend(frameon=False, fontsize=7); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(100 * max((m[1] or 0), m[2]) for m in ms) * 1.18 + 1)
    paths = save_figure(fig, rep.dir, "fig_webcam_vs_iam")
    rep.add(paths, "Character error rate on IAM test words vs on the webcam test set, for each model (domain gap from "
                   "scanned forms to phone-camera photos).")
    return paths


def save_prediction_grid(rep, rows, hyps, n=12, seed=0, name="fig_webcam_predictions", model="CRNN Model A"):
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    imgs = [rows[i]["clean"] if rows[i]["clean"] is not None else np.full((32, 64), 250, np.uint8) for i in idx]
    refs, hy = [rows[i]["label"] for i in idx], [hyps[i] for i in idx]
    paths = figures.plot_prediction_grid(imgs, refs, hy, rep.dir, name)
    right = sum(r == h for r, h in zip(refs, hy))
    rep.add(paths, f"{model} on {len(idx)} randomly chosen webcam words ({right}/{len(idx)} exactly right), shown "
                   "after the photo clean-up; green = correct, red = wrong.")
    return paths


def save_preprocessing_figure(rep, rows, rec, n=4, seed=1):
    """Raw guide-box crop -> cleaned word -> the 32-px model input, for a few random words."""
    import cv2
    import matplotlib.pyplot as plt
    from src.demo.webcam import tensor_to_u8
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    fig, axes = plt.subplots(len(idx), 3, figsize=(FULL_WIDTH, 0.75 * len(idx) + 0.3),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]}, squeeze=False)
    for (a, b, c), i in zip(axes, idx):
        r = rows[i]
        a.imshow(cv2.cvtColor(r["raw"], cv2.COLOR_BGR2RGB)); a.set_ylabel(r["label"], rotation=0, ha="right",
                                                                         va="center", fontsize=7.5)
        if r["clean"] is not None:
            b.imshow(r["clean"], cmap="gray", vmin=0, vmax=255)
            c.imshow(tensor_to_u8(rec.tensor(r["clean"])), cmap="gray", vmin=0, vmax=255)
        for ax in (a, b, c):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax, t in zip(axes[0], ("camera crop (guide box)", "after clean-up", "model input (32 px high)")):
        ax.set_title(t, fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    paths = save_figure(fig, rep.dir, "fig_webcam_preprocessing")
    rep.add(paths, "Webcam preprocessing: the camera crop, after the photo clean-up (lighting flattened, ink found, "
                   "tight crop, IAM-like contrast), and the 32-px input the CRNN reads (same transform as training).")
    return paths

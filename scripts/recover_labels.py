"""Rebuild the missing lines of a cut-off words.txt.

Some IAM copies (e.g. the Kaggle one) have every word image but a words.txt that
stops part-way (at e07-072). This script writes words_recovered.txt: all the
original lines, plus a rebuilt line for every image that had no label.

Where the labels come from (see src/data/recover.py):
  1. the IAM XML embedded in your own PNGs, if they still have it (no download), else
  2. the Hugging Face copy of IAM (priyank-m/IAM_words_text_recognition, ~1.2 GB
     parquet), whose original PNGs carry the same XML. Entries are matched by
     word ID. Any image whose ID isn't found is matched by a hash of its pixels.

Checks printed for the report:
  - rebuilt vs original lines for words that have both (should be ~100% identical)
  - pixel check: HF image == your image for a sample of recovered words
  - matched / unmatched / ambiguous counts. Below --min-rate (90%) nothing is
    written, and you keep training on the labelled subset.

  python scripts/recover_labels.py --data-root /content/iam --out words_recovered.txt
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.data.iam import image_path, parse_words_txt
from src.data.recover import entry_from_png_bytes, entry_from_png_path, pixel_hash, words_txt_line

HF_REPO = "priyank-m/IAM_words_text_recognition"
HF_FILES = ["data/train-00000-of-00001-bfc7b63751c36ab0.parquet",
            "data/val-00000-of-00001-472467affea948eb.parquet",
            "data/test-00000-of-00001-4cae70e6a03872e2.parquet"]


def list_image_ids(words_dir: Path) -> dict:
    """{word_id: path} for every PNG under words/ (follows the setup symlinks)."""
    import os
    ids = {}
    for d, _, files in os.walk(words_dir, followlinks=True):
        for f in files:
            if f.endswith(".png"):
                ids[f[:-4]] = Path(d) / f
    return ids


def download_hf(cache_dir: Path) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        from huggingface_hub import hf_hub_download
        for f in HF_FILES:
            paths.append(Path(hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=str(cache_dir))))
        return paths
    except ImportError:
        pass
    for f in HF_FILES:
        p = cache_dir / Path(f).name
        if not p.exists():
            print(f"  downloading {f} ...")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{f}", p)
        paths.append(p)
    return paths


def scan_hf(parquets, wanted: set, sample_ids: set, threads=8):
    """One pass over the HF parquet files.
    Returns ({id: entry} for wanted ids, {id: png bytes} for sample ids, rows without an id)."""
    import pyarrow.parquet as pq
    found, sample_bytes, no_id = {}, {}, []
    for p in parquets:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2000, columns=["image", "text"]):
            imgs = batch.column("image").to_pylist()
            texts = batch.column("text").to_pylist()
            with ThreadPoolExecutor(threads) as pool:
                entries = list(pool.map(lambda d: entry_from_png_bytes(d["bytes"]), imgs))
            for e, d, t in zip(entries, imgs, texts):
                if e is None:
                    no_id.append((d["bytes"], t))
                    continue
                if e["id"] in wanted:
                    found[e["id"]] = e
                if e["id"] in sample_ids:
                    sample_bytes[e["id"]] = d["bytes"]
        print(f"  scanned {Path(p).name}: {len(found)} wanted ids found so far")
    return found, sample_bytes, no_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder with ascii/words.txt and words/")
    ap.add_argument("--words-txt", default=None, help="default: <data-root>/ascii/words.txt")
    ap.add_argument("--out", required=True, help="where to write words_recovered.txt")
    ap.add_argument("--hf-cache", default="/content/hf_iam", help="where to put the HF parquet files")
    ap.add_argument("--hf-parquet", nargs="*", default=None, help="use these local parquet files")
    ap.add_argument("--no-local-xml", action="store_true", help="skip reading XML from your own PNGs")
    ap.add_argument("--min-rate", type=float, default=0.90)
    ap.add_argument("--verify-sample", type=int, default=3000)
    ap.add_argument("--report", default=None, help="also save the numbers as JSON here")
    ap.add_argument("--preview", default=None, help="save a picture of recovered words here")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    root = Path(a.data_root)
    words_txt = Path(a.words_txt or root / "ascii" / "words.txt")

    malformed = []
    original = {wid: (st, txt) for wid, st, txt in parse_words_txt(words_txt, malformed)}
    original_lines = [l.rstrip("\n") for l in open(words_txt, encoding="utf-8")
                      if l.strip() and not l.startswith("#") and len(l.split(" ")) >= 9]
    images = list_image_ids(root / "words")
    unlabeled = sorted(set(images) - set(original))
    print(f"words.txt: {len(original)} labelled words ({len(malformed)} cut-off line(s)); "
          f"images: {len(images)}; images without a label: {len(unlabeled)}")
    report = {"labelled_original": len(original), "images": len(images), "unlabeled": len(unlabeled)}
    if not unlabeled:
        print("Nothing to recover: every image already has a label.")
        return

    recovered, source = {}, None
    # ---- 1. XML inside the user's own PNGs (free, no download)
    if not a.no_local_xml:
        probe = unlabeled[:200]
        with ThreadPoolExecutor(a.threads) as pool:
            hits = [e for e in pool.map(lambda i: entry_from_png_path(images[i]), probe) if e]
        print(f"embedded XML in your own images: {len(hits)}/{len(probe)} probed")
        if len(hits) >= 0.9 * len(probe):
            with ThreadPoolExecutor(a.threads) as pool:
                for wid, e in zip(unlabeled, pool.map(lambda i: entry_from_png_path(images[i]), unlabeled)):
                    if e and e["id"] == wid:
                        recovered[wid] = e
            source = "embedded XML in the user's PNGs"

    # ---- 2. Hugging Face copy of IAM, matched by word ID (pixel hash as backup)
    sample_ids = set(random.Random(0).sample(unlabeled, min(a.verify_sample, len(unlabeled))))
    hf_sample = {}
    if len(recovered) < a.min_rate * len(unlabeled):
        recovered.clear()
        parquets = [Path(p) for p in a.hf_parquet] if a.hf_parquet else download_hf(Path(a.hf_cache))
        # also fetch the originally-labelled ids, to check rebuilt lines against real ones
        check_ids = set(random.Random(1).sample(sorted(original), min(5000, len(original))))
        found, hf_sample, no_id = scan_hf(parquets, set(unlabeled) | check_ids, sample_ids, a.threads)
        recovered = {i: found[i] for i in unlabeled if i in found}
        source = f"Hugging Face {HF_REPO} (matched by word ID)"

        # check rebuilt lines against the original words.txt lines
        orig_line = {l.split(" ")[0]: l for l in original_lines}
        both = [i for i in check_ids if i in found and i in orig_line]
        same = sum(words_txt_line(found[i]) == orig_line[i] for i in both)
        same_label = sum(found[i]["text"] == original[i][1] and found[i]["status"] == original[i][0] for i in both)
        print(f"rebuilt vs original words.txt ({len(both)} words): identical lines {same/len(both):.2%}, "
              f"same text+status {same_label/len(both):.2%}" if both else "no overlap to check")
        report.update(check_words=len(both), identical_lines=same, same_text_status=same_label)

        # backup: pixel-hash match for images whose id wasn't found (expected: none)
        missing = [i for i in unlabeled if i not in recovered]
        if missing and no_id:
            print(f"{len(missing)} ids not found; trying pixel hashes against {len(no_id)} HF rows without ids")
            by_hash = {}
            for data, text in no_id:
                h = pixel_hash(Image.open(__import__("io").BytesIO(data)))
                by_hash.setdefault(h, []).append(text)
            amb = 0
            for i in missing:
                texts = by_hash.get(pixel_hash(Image.open(images[i])), [])
                if len(set(texts)) == 1:
                    recovered[i] = {"id": i, "status": "ok", "threshold": "0", "box": (-1, -1, -1, -1),
                                    "tag": "RECOVERED", "text": texts[0]}
                elif texts:
                    amb += 1
            report["ambiguous"] = amb

    # ---- pixel check: is the HF image really the same picture as ours?
    if hf_sample:
        import io
        def same(i):
            try:
                return pixel_hash(Image.open(io.BytesIO(hf_sample[i]))) == pixel_hash(Image.open(images[i]))
            except Exception:
                return False
        with ThreadPoolExecutor(a.threads) as pool:
            res = list(pool.map(same, sorted(hf_sample)))
        rate = sum(res) / len(res)
        print(f"pixel check on {len(res)} recovered words: {rate:.2%} identical images")
        report.update(pixel_checked=len(res), pixel_identical=sum(res))

    n = len(recovered)
    rate = n / len(unlabeled)
    report.update(source=source, matched=n, unmatched=len(unlabeled) - n, match_rate=round(rate, 4),
                  ambiguous=report.get("ambiguous", 0))
    print(f"\nmatched {n} / {len(unlabeled)} unlabeled images ({rate:.2%}); "
          f"unmatched {len(unlabeled) - n}; ambiguous {report['ambiguous']}; source: {source}")
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2))

    if rate < a.min_rate:
        print(f"\nMatch rate is below {a.min_rate:.0%}, so NO words_recovered.txt was written. "
              "Training will use the labelled subset only.")
        sys.exit(2)

    out = Path(a.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write("#--- words_recovered.txt: IAM words.txt format ---\n"
                f"# original lines from {words_txt.name}, then {n} lines rebuilt from IAM's embedded XML\n"
                f"# source of rebuilt lines: {source}\n"
                "# format: a01-000u-00-00 ok 154 408 768 27 51 AT A\n")
        f.write("\n".join(original_lines) + "\n")
        f.write("#--- recovered entries below (one per image that had no label) ---\n")
        f.write("\n".join(words_txt_line(recovered[i]) for i in sorted(recovered)) + "\n")
    out.with_name(out.stem + "_ids.txt").write_text("\n".join(sorted(recovered)) + "\n")
    print(f"wrote {out} ({len(original_lines) + n} words) in {time.time() - t0:.0f}s")

    if a.preview:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = random.Random(2).sample(sorted(recovered), min(12, n))
        fig, axes = plt.subplots(len(pick), 1, figsize=(5, 0.9 * len(pick) + 0.4))
        for ax, i in zip(axes if len(pick) > 1 else [axes], pick):
            ax.imshow(Image.open(images[i]).convert("L"), cmap="gray")
            ax.set_title(f"{i}: {recovered[i]['text']!r} ({recovered[i]['status']})", fontsize=8, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(a.preview, dpi=110); plt.close(fig)
        print(f"saved preview of recovered labels to {a.preview}")


if __name__ == "__main__":
    main()

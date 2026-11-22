"""Tesseract OCR baseline: an off-the-shelf recognizer run on the same IAM test words.

Settings (also in the report):
  --oem 1   use Tesseract's LSTM recognizer only (not the legacy engine)
  --psm 8   "treat the image as a single word" (IAM crops are single words)
  -l eng    English model shipped with tesseract-ocr

Preprocessing is deliberately light, so the baseline is fair but not tuned:
  grayscale + a white border of `border` px on every side. Tesseract is trained on
  text with margins; IAM crops touch the ink, and without a border it often returns nothing.
  No resizing, binarization or deskewing (Tesseract binarizes internally).

Output normalization: strip whitespace/newlines (psm 8 still sometimes splits a word) and
turn curly quotes into the straight ASCII quotes IAM uses. Nothing else is corrected.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

OEM, PSM, BORDER = 1, 8, 16
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-"})


def preprocess(img: Image.Image, border: int = BORDER) -> Image.Image:
    return ImageOps.expand(img.convert("L"), border=border, fill=255)


def normalize(text: str) -> str:
    return "".join(text.split()).translate(_QUOTES)


def config(oem=OEM, psm=PSM) -> str:
    return f"--oem {oem} --psm {psm}"


def load_image(path):
    """The word image, or None if the file is missing, empty or can't be decoded."""
    try:
        with Image.open(path) as im:
            im.load()
            return im.copy() if im.size[0] and im.size[1] else None
    except Exception:
        return None


def recognize(path, border=BORDER, oem=OEM, psm=PSM, lang="eng"):
    """Tesseract's normalized output, or None if the image can't be read (the word is then skipped)."""
    import pytesseract
    img = load_image(path)
    if img is None:
        return None
    return normalize(pytesseract.image_to_string(preprocess(img, border), lang=lang, config=config(oem, psm)))


def version() -> str:
    import pytesseract
    return str(pytesseract.get_tesseract_version())


def run(items, workers=None, border=BORDER, progress=True):
    """Recognize every item (IAM WordItem). Returns (predictions, seconds).

    Each call starts one tesseract process; we run `workers` of them in parallel (one per CPU
    core), each limited to one thread (OMP_THREAD_LIMIT=1) so they don't fight over cores."""
    workers = workers or os.cpu_count() or 2
    t0, out, done = time.time(), [None] * len(items), 0
    step = max(1, len(items) // 10)
    # Only the tesseract child processes should see this: PyTorch (OpenMP) can hang with it.
    old = os.environ.get("OMP_THREAD_LIMIT")
    os.environ["OMP_THREAD_LIMIT"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(recognize, it.path, border): i for i, it in enumerate(items)}
            for f in futures:
                out[futures[f]] = f.result()
                done += 1
                if progress and done % step == 0:
                    el = time.time() - t0
                    print(f"  {done:,}/{len(items):,} words  {el:.0f}s elapsed, ~{el / done * (len(items) - done):.0f}s left")
    finally:
        if old is None:
            os.environ.pop("OMP_THREAD_LIMIT", None)
        else:
            os.environ["OMP_THREAD_LIMIT"] = old
    return out, time.time() - t0


FIELDS = ["word_id", "writer", "ref", "hyp", "correct", "ms"]


def run_cached(items, cache_csv, workers=None, chunk=500, border=BORDER):
    """Like run(), but saves results to `cache_csv` every `chunk` words and skips words already
    there, so a Colab disconnect only loses the current chunk. Returns (rows in `items` order, seconds)
    where `ms` per row is the measured wall-clock time per word of its chunk."""
    import csv
    done = {}
    if os.path.exists(cache_csv):
        with open(cache_csv, newline="") as f:
            done = {r["word_id"]: r for r in csv.DictReader(f)}
    todo = [it for it in items if it.word_id not in done]
    if done:
        print(f"  reusing {len(items) - len(todo):,} cached predictions from {os.path.basename(cache_csv)}")
    os.makedirs(os.path.dirname(os.path.abspath(cache_csv)), exist_ok=True)
    new_file = not os.path.exists(cache_csv)
    t_all, skipped = time.time(), []
    with open(cache_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for start in range(0, len(todo), chunk):
            part = todo[start:start + chunk]
            hyps, secs = run(part, workers=workers, border=border, progress=False)
            ms = 1000 * secs / len(part)
            for it, h in zip(part, hyps):
                if h is None:                       # unreadable image: skipped, not cached
                    skipped.append(it.word_id)
                    continue
                row = {"word_id": it.word_id, "writer": it.writer or "", "ref": it.text, "hyp": h,
                       "correct": str(int(h == it.text)), "ms": f"{ms:.2f}"}
                done[it.word_id] = row
                w.writerow(row)
            f.flush()
            el = time.time() - t_all
            n = start + len(part)
            print(f"  {n:,}/{len(todo):,} new words, {el:.0f}s elapsed, ~{el / n * (len(todo) - n):.0f}s left")
    if skipped:
        print(f"  skipped {len(skipped)} unreadable image(s): {', '.join(skipped[:10])}")
    rows = [done[it.word_id] for it in items if it.word_id in done]
    return rows, sum(float(r["ms"]) for r in rows) / 1000

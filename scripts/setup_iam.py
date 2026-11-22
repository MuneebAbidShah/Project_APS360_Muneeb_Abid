"""Turn whatever IAM download you have into the layout the code expects.

Target layout (DATA_ROOT):
    <out>/ascii/words.txt          (+ forms.txt if your download has it)
    <out>/words/a01/a01-000u/a01-000u-00-00.png

Two ways to use it:

1) Auto-detect inside an unpacked folder (official or Kaggle download):
       python scripts/setup_iam.py --raw /content/iam_raw --out /content/iam
   It unpacks any .tgz / .tar.gz it finds, locates words.txt, and finds the
   folder of word images. It links them into <out> with symlinks, so nothing
   is copied. Images stored in one flat folder also work: each gets its own link.

2) The two official archives:
       python scripts/setup_iam.py --words-tgz words.tgz --ascii-tgz ascii.tgz --out data/iam
"""
import argparse
import os
import re
import shutil
import tarfile
from pathlib import Path

# IAM word IDs look like a01-000u-00-00 (form prefix, form, line, word)
WORD_ID_RE = re.compile(r"^[a-z]\d\d-\d{3}[a-z]?-\d\d-\d\d$")


# ------------------------------------------------------------ helpers

def safe_extract(tgz: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(dest, filter="data")   # safe mode (Python 3.12+)
        except TypeError:
            tf.extractall(dest)                  # older Python


def extract_nested_archives(raw: Path):
    """Unpack every .tgz/.tar.gz under `raw` once (a marker file remembers)."""
    archives = [p for p in raw.rglob("*") if p.is_file() and p.name.endswith((".tgz", ".tar.gz"))]
    for a in archives:
        dest = a.parent / (a.name.split(".")[0] + "_extracted")
        if (dest / ".done").exists():
            continue
        print(f"  extracting {a} -> {dest}")
        safe_extract(a, dest)
        (dest / ".done").touch()


def count_word_lines(path: Path, max_lines=2000) -> int:
    """How many of the first lines look like IAM words.txt entries."""
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.split(" ")
            if len(parts) >= 9 and WORD_ID_RE.match(parts[0]) and parts[1] in ("ok", "err"):
                n += 1
    return n


def find_words_txt(raw: Path) -> Path:
    candidates = [p for p in raw.rglob("*.txt") if p.is_file()]
    scored = sorted(((count_word_lines(p), p) for p in candidates), reverse=True)
    if not scored or scored[0][0] == 0:
        raise FileNotFoundError(f"No IAM-style words.txt found under {raw}")
    # prefer a file literally called words.txt among equally good ones
    best_score = scored[0][0]
    best = [p for s, p in scored if s == best_score]
    return next((p for p in best if p.name == "words.txt"), best[0])


def find_image_root(raw: Path):
    """Return ('nested', root) if images sit at root/a01/a01-000u/<id>.png,
    else ('flat', None) when they don't follow that folder structure."""
    for png in raw.rglob("*.png"):
        wid = png.stem
        if not WORD_ID_RE.match(wid):
            continue
        form = "-".join(wid.split("-")[:2])
        if png.parent.name == form and png.parent.parent.name == form.split("-")[0]:
            return "nested", png.parent.parent.parent
        return "flat", None
    raise FileNotFoundError(f"No IAM word images (e.g. a01-000u-00-00.png) found under {raw}")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src.resolve(), dst)


# ------------------------------------------------------------ main modes

def setup_from_raw(raw: Path, out: Path, words_txt: Path | None = None):
    raw = Path(raw)
    extract_nested_archives(raw)

    # A words.txt you pass in (e.g. the full official one) wins over the one in the download.
    words_txt = Path(words_txt) if words_txt else find_words_txt(raw)
    print(f"  words.txt   : {words_txt}")
    link(words_txt, out / "ascii" / "words.txt")
    forms = [p for p in raw.rglob("forms.txt") if p.is_file()]
    if forms:
        print(f"  forms.txt   : {forms[0]}")
        link(forms[0], out / "ascii" / "forms.txt")
    else:
        print("  forms.txt   : not found (fine: the Aachen splits don't need it)")

    kind, root = find_image_root(raw)
    if kind == "nested":
        print(f"  image root  : {root} (nested a01/a01-000u/... layout)")
        link(root, out / "words")
    else:
        print("  images are not in a01/a01-000u/ folders: linking each one into place")
        n = 0
        for png in raw.rglob("*.png"):
            wid = png.stem
            if WORD_ID_RE.match(wid):
                form = "-".join(wid.split("-")[:2])
                link(png, out / "words" / form.split("-")[0] / form / png.name)
                n += 1
        print(f"  linked {n} images")


def setup_from_tgz(words_tgz: Path, ascii_tgz: Path, out: Path):
    raw = out.parent / (out.name + "_raw")
    safe_extract(Path(ascii_tgz), raw / "ascii")
    safe_extract(Path(words_tgz), raw / "words")
    setup_from_raw(raw, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="folder with an unpacked IAM download (any layout)")
    ap.add_argument("--words-tgz")
    ap.add_argument("--ascii-tgz")
    ap.add_argument("--words-txt", help="use this words.txt instead of the one in the download")
    ap.add_argument("--out", default="data/iam")
    a = ap.parse_args()
    out = Path(a.out)
    if a.raw:
        setup_from_raw(Path(a.raw), out, a.words_txt)
    elif a.words_tgz and a.ascii_tgz:
        setup_from_tgz(Path(a.words_tgz), Path(a.ascii_tgz), out)
    else:
        ap.error("give --raw, or both --words-tgz and --ascii-tgz")
    # os.walk with followlinks=True also counts images behind the symlinks
    n = sum(len([f for f in fs if f.endswith(".png")]) for _, _, fs in os.walk(out / "words", followlinks=True))
    print(f"IAM ready in {out.resolve()}: {n} word images")
    check_words_txt(out / "ascii" / "words.txt", n)


def check_words_txt(words_txt: Path, n_images: int):
    """Warn if words.txt labels far fewer words than there are images
    (e.g. a words.txt that was cut off part-way through)."""
    n_labels, last = 0, ""
    with open(words_txt, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                n_labels += 1
                last = line.rstrip("\n")
    print(f"words.txt lists {n_labels} words; last line: {last!r}")
    if len(last.split(" ")) < 9:
        print("WARNING: the last line of words.txt is incomplete, so the file was cut off. "
              "Words after it have no labels and will be skipped.")
    if n_images > 1.1 * n_labels:
        print(f"WARNING: {n_images - n_labels} images have no line in words.txt. "
              "Get the full words.txt (inside IAM's official ascii.tgz) to use all of them.")


if __name__ == "__main__":
    main()

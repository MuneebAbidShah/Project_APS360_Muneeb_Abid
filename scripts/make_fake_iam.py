"""Make a small fake IAM dataset so the pipeline can run before you have IAM.

It renders words with fonts (handwriting-style ones if your system has them)
into exactly the IAM layout:

    <out>/ascii/words.txt
    <out>/ascii/forms.txt
    <out>/words/a01/a01-000u/a01-000u-00-00.png

Each fake "writer" uses one font. A few deliberately bad samples are added
(an 'err' status, non-charset characters, an over-long word, an empty image
file) so you can see every filter in inspect_data.py drop something.

Usage:  python scripts/make_fake_iam.py --out data/fake_iam
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from xml.sax.saxutils import quoteattr

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORDS = (
    "the of and to a in that is was he for it with as his on be at by I had "
    "not are but from or have an they which one you were her all she there "
    "would their we him been has when who will more no if out so said what up "
    "its about into than them can only other new some could time these two may "
    "then do first any my now such like our over man me even most made after "
    "also did many before must through back years where much your way well down "
    "should because each just those people Mr how too little state good very "
    "make world still own see men work long get here between both life being "
    "under never day same another know while last might us great old year off "
    "come since against go came right used take three London Labour Kennedy "
    "President Government British House Sunday"
).split() + [",", ".", "'s", "-", "?", "!", "\"", "1960", "42", "(it)", "don't", "e.g."]

FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"),
    "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts",
]
HANDWRITING_HINTS = ("caveat", "kalam", "patrick", "indie", "shadows", "handlee", "gochi",
                     "homemade", "dekko", "gloria", "architect", "coming", "nanumpen",
                     "justanother", "rocksalt", "dancing", "satisfy", "marker", "mali")


def find_fonts(extra_dir=None, limit=12):
    """Return up to `limit` .ttf/.otf paths, preferring handwriting-like fonts."""
    dirs = ([extra_dir] if extra_dir else []) + FONT_DIRS
    fonts = []
    for d in dirs:
        p = Path(d)
        if p.exists():
            fonts += [f for f in p.rglob("*") if f.suffix.lower() in (".ttf", ".otf")]
    hand = [f for f in fonts if any(h in f.name.lower().replace(" ", "") for h in HANDWRITING_HINTS)]
    picked = sorted(set(hand))[:limit]
    if len(picked) < limit:  # top up with any other fonts
        others = sorted(set(fonts) - set(picked))
        random.Random(0).shuffle(others)
        picked += others[: limit - len(picked)]
    return picked


def load_font(path, size):
    if path is None:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    return ImageFont.truetype(str(path), size)


def render_word(text, font, rng):
    """Dark ink on light, slightly noisy paper, cropped tightly like IAM."""
    l, t, r, b = font.getbbox(text)
    pad = 6
    w, h = max(r - l, 1) + 2 * pad, max(b - t, 1) + 2 * pad
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, fill=rng.randint(0, 70), font=font)
    img = img.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * rng.uniform(0.88, 1.0) + np.random.default_rng(rng.randint(0, 10**9)).normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fake_iam_xml(form_id, writer_id, word_id, status, text, box):
    """Minimal version of the XML IAM stores inside every word PNG."""
    x, y, w, h = box
    q = quoteattr
    return (f'<?xml version="1.0"?>\n<!DOCTYPE form SYSTEM \'form-metadata.dtd\'>\n'
            f'<form id={q(form_id)} writer-id={q(writer_id)} '
            f'select="descendant::word[@id=&apos;{word_id}&apos;]"><handwritten-part>'
            f'<line id={q(word_id.rsplit("-", 1)[0])} segmentation={q(status)} threshold="180">'
            f'<word id={q(word_id)} tag="NN" text={q(text)}>'
            f'<cmp x="{x}" y="{y}" width="{w}" height="{h}" /></word></line>'
            f'</handwritten-part></form>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_iam")
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--forms-per-writer", type=int, default=2)
    ap.add_argument("--lines-per-form", type=int, default=4)
    ap.add_argument("--words-per-line", type=int, default=6)
    ap.add_argument("--font-dir", default=None, help="extra folder to search for fonts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-xml", action="store_true",
                    help="store IAM-style XML metadata inside each PNG, like the real IAM files "
                         "(lets scripts/recover_labels.py be tested)")
    ap.add_argument("--real-form-ids", action="store_true",
                    help="use real IAM form IDs from the Aachen lists (and their real writers), "
                         "so the Aachen split and writer table code paths get exercised")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "ascii").mkdir(parents=True, exist_ok=True)
    fonts = find_fonts(args.font_dir, args.writers) or [None]
    print(f"Using {len(fonts)} font(s): {[f.name if f else 'PIL default' for f in fonts]}")

    words_lines = ["#--- fake IAM words.txt (same format as the real one) ---",
                   "# format: a01-000u-00-00 ok 154 408 768 27 51 AT A"]
    forms_lines = ["#--- fake IAM forms.txt ---", "# format: a01-000u 000 2 prt 7 5 52 36"]
    # Build the list of (form_id, writer_id) to generate.
    forms = []
    if args.real_form_ids:
        root = Path(__file__).resolve().parents[1] / "data"
        table = dict(l.strip().split(",") for l in open(root / "iam_form_writers.csv") if "," in l and not l.startswith("form_id"))
        n_per = {"train.uttlist": args.writers * args.forms_per_writer * 6 // 10}
        n_per["validation.uttlist"] = n_per["test.uttlist"] = max(1, args.writers * args.forms_per_writer * 2 // 10)
        for fname, k in n_per.items():
            ids = [l.strip() for l in open(root / "splits" / "aachen" / fname) if l.strip()][:k]
            forms += [(f, table[f]) for f in ids]
    else:
        # 'z' prefixes never occur in real IAM, so the Aachen split lists won't
        # match and the automatic writer-disjoint split is used instead.
        for wi in range(args.writers):
            for fi in range(args.forms_per_writer):
                forms.append((f"z{wi + 1:02d}-{fi * 10:03d}{'u' if fi % 2 == 0 else 'x'}", f"{wi:03d}"))

    bad_done = set()
    n = 0
    writer_index = {w: i for i, w in enumerate(sorted({w for _, w in forms}))}
    for form_id, writer_id in forms:
        font_path = fonts[writer_index[writer_id] % len(fonts)]   # one font per writer
        prefix = form_id.split("-")[0]
        forms_lines.append(f"{form_id} {writer_id} 2 prt 7 5 52 36")
        form_dir = out / "words" / prefix / form_id
        form_dir.mkdir(parents=True, exist_ok=True)
        for li in range(args.lines_per_form):
            for k in range(args.words_per_line):
                word_id = f"{form_id}-{li:02d}-{k:02d}"
                text, status = rng.choice(WORDS), "ok"
                # Sprinkle in one example of each kind of bad data.
                if "err" not in bad_done and n == 5:
                    status = "err"; bad_done.add("err")
                elif "char" not in bad_done and n == 11:
                    text = "café"; bad_done.add("char")
                elif "long" not in bad_done and n == 17:
                    text = "antidisestablishmentarianismlongword"; bad_done.add("long")
                font = load_font(font_path, rng.randint(36, 60))
                path = form_dir / f"{word_id}.png"
                if "empty" not in bad_done and n == 23:
                    path.write_bytes(b""); bad_done.add("empty")
                    w, h = 10, 10
                else:
                    img = render_word(text, font, rng)
                    w, h = img.size
                    meta = None
                    if args.embed_xml:
                        meta = PngInfo()
                        meta.add_text("xml", fake_iam_xml(form_id, writer_id, word_id, status, text,
                                                          (k * 200, li * 100, w, h)))
                    img.save(path, pnginfo=meta)
                words_lines.append(f"{word_id} {status} 180 {k * 200} {li * 100} {w} {h} NN {text}")
                n += 1

    (out / "ascii" / "words.txt").write_text("\n".join(words_lines) + "\n", encoding="utf-8")
    (out / "ascii" / "forms.txt").write_text("\n".join(forms_lines) + "\n", encoding="utf-8")
    print(f"Wrote {n} word images to {out.resolve()}")


if __name__ == "__main__":
    main()

"""Rule-based letter splitter for IAM word images + EMNIST-style normalization + word decoding.

Pipeline for one word image (all rules, no learning):
  1. binarize: light blur + Otsu threshold (ink = True); then **slant correction**: shear the word
     so its strokes are vertical (the shear that maximizes the sum of squared column ink counts),
     otherwise slanted letters overlap and vertical cuts slice through them;
  2. connected components (8-connected), then merge components that overlap horizontally
     (the dot of an i/j, broken strokes) into one group;
  3. estimate the pen's stroke width (2 x ink area / ink perimeter); the expected character width is
     `width_factor` x stroke width (on IAM training words, width / characters / stroke width has a
     median of ~7 and varies less between writers than the x-height); the core (x-)height band,
     from the horizontal ink profile, is used for the punctuation rules;
  4. small groups are punctuation candidates (see `_punct`); everything else is a letter group;
  5. a letter group wider than `split_ratio` expected widths is cut into n = round(width / expected
     width) pieces at the columns with the least ink (vertical projection profile) near the even cuts;
  6. pieces narrower than `min_piece` expected widths are merged into their narrower neighbour;
  7. each piece is normalized like EMNIST: cropped to its ink, stroke thickness adjusted, scaled so
     the long side is 24 px, centered in 28x28 (2-px border), white ink on black, slightly blurred.

Characters EMNIST doesn't have: EMNIST only knows 0-9, A-Z, a-z. Small marks are labelled by
position/shape rules instead: '.' (dot on the baseline), ',' (taller mark at the baseline),
"'" (mark above the letters), '"' (two marks above), '-' (flat mark in the middle), and for
single-mark word images also ':' ';' '!'. Other symbols (#&()*+/?) are never output, so they
always count as errors: this is a limitation of the baseline, not hidden.
Scoring is case-sensitive, exactly like the CRNN and Tesseract.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from src.baselines.emnist import CLASSES


@dataclass
class SegParams:
    width_factor: float = 8.0    # expected character width = width_factor x stroke width
    split_ratio: float = 1.45    # only split groups wider than this many expected widths
    min_piece: float = 0.3       # merge pieces narrower than this many expected widths
    overlap_merge: float = 0.45  # merge components overlapping by this fraction of the narrower one
    noise: float = 0.012         # drop components with area < noise x core height^2 (and < 12 px)


def load_gray(path):
    """uint8 grayscale image, or None if the file is missing, empty or can't be decoded."""
    try:
        with Image.open(path) as im:
            im.load()
            a = np.asarray(im.convert("L"))
        return a if a.size and min(a.shape) > 0 else None
    except Exception:
        return None


def binarize(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    if int(g.max()) - int(g.min()) < 30:          # blank / flat image: no ink
        return np.zeros(gray.shape, bool)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


SHEARS = np.round(np.arange(-0.8, 0.81, 0.1), 2)


def _shear(img, s, fill):
    """x' = x + s * (y - H/2) (positive s straightens right-leaning writing)."""
    H, W = img.shape
    pad = int(np.ceil(abs(s) * H / 2)) + 1
    M = np.float32([[1, s, pad - s * H / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (W + 2 * pad, H), flags=cv2.INTER_NEAREST, borderValue=fill)


def deslant(mask, shears=SHEARS):
    """Return (sheared mask, shear) with the most vertical strokes."""
    if not mask.any():
        return mask, 0.0
    m8 = mask.astype(np.uint8)
    best, bs = -1.0, 0.0
    for s in shears:
        col = _shear(m8, s, 0).sum(0).astype(np.float64)
        score = (col ** 2).sum()
        if score > best:
            best, bs = score, float(s)
    return (_shear(m8, bs, 0) > 0) if bs else mask, bs


def _core(mask):
    """(top, bottom, height) of the core band: the rows with at least half the peak row ink."""
    rows = mask.sum(1).astype(float)
    if rows.max() == 0:
        return 0, mask.shape[0], mask.shape[0]
    on = rows >= 0.5 * rows.max()
    best, cur, start, bs = 0, 0, 0, 0
    for i, v in enumerate(on):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bs = cur, start
        else:
            cur = 0
    ink_rows = np.nonzero(rows)[0]
    h = max(best, 0.25 * (ink_rows[-1] - ink_rows[0] + 1), 4)
    return bs, bs + best, h


class Group:
    __slots__ = ("x0", "x1", "y0", "y1", "labels", "kind", "char", "area")

    def __init__(self, x0, x1, y0, y1, labels, area):
        self.x0, self.x1, self.y0, self.y1, self.labels, self.area = x0, x1, y0, y1, set(labels), area
        self.kind, self.char = "char", None

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def merge(self, o):
        self.x0, self.x1 = min(self.x0, o.x0), max(self.x1, o.x1)
        self.y0, self.y1 = min(self.y0, o.y0), max(self.y1, o.y1)
        self.labels |= o.labels; self.area += o.area


def _single_mark(groups, H, W):
    """Rules for a word image that is only punctuation (IAM crops are tight, so use absolute sizes;
    IAM is scanned at 300 dpi: a period is ~8 px, a lower-case letter ~30-40 px high)."""
    if W > 60 or H > 70:
        return None
    if len(groups) == 1:
        g = groups[0]
        if g.h <= 24 and g.w <= 26 and 0.5 <= g.h / max(g.w, 1) <= 2.0:
            return "."
        if g.h <= 16 and g.w >= 1.8 * g.h:
            return "-"
        if g.h <= 45 and g.w <= 30 and g.h >= 1.4 * g.w:
            return ","
        return None
    if len(groups) == 2:
        a, b = sorted(groups, key=lambda g: g.y0)
        stacked = min(a.x1, b.x1) - max(a.x0, b.x0) > 0 or abs((a.x0 + a.x1) - (b.x0 + b.x1)) < max(a.w, b.w)
        side = not stacked and abs(a.y0 - b.y0) < 0.5 * max(a.h, b.h)
        if side and max(a.h, b.h) <= 45:
            return '"'
        if stacked and b.y0 >= a.y1 - 2:
            small = lambda g: g.h <= 24 and g.w <= 26
            if small(a) and small(b):
                return ":"
            if small(a) and b.h >= 1.4 * b.w and b.h <= 45:
                return ";"
            if small(b) and a.h >= 2 * a.w:
                return "!"
    return None


def _punct(g, core_top, core_bot, xh):
    """Label a small group inside a longer word by its size and position relative to the core band."""
    if g.h > 0.6 * xh or g.w > 0.8 * xh:
        if g.h <= 0.35 * xh and g.w >= 1.8 * g.h and core_top - 0.1 * xh <= (g.y0 + g.y1) / 2 <= core_bot:
            return "-"
        return None
    cy = (g.y0 + g.y1) / 2
    if g.y1 <= core_top + 0.2 * xh:
        return "'"
    if cy >= core_bot - 0.25 * xh:
        return "." if g.h <= 1.3 * max(g.w, 1) else ","
    if g.w >= 1.8 * g.h:
        return "-"
    return None


def segment(gray, p: SegParams = SegParams()):
    """Split a word image into character pieces. Returns (pieces, info); each piece is a dict
    with x0, x1, y0, y1, kind ('char' or 'punct'), char (for punct) and mask (bool crop)."""
    mask = binarize(gray)
    mask, shear = deslant(mask)
    H, W = mask.shape
    if not mask.any():
        return [], {"core": (0, H, H), "shear": 0.0}
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ct, cb, xh = _core(mask)
    min_area = min(12, p.noise * xh * xh)
    groups = [Group(s[0], s[0] + s[2], s[1], s[1] + s[3], [i], s[4])
              for i, s in enumerate(stats) if i > 0 and s[4] >= max(min_area, 2)]
    if not groups:
        return [], {"core": (ct, cb, xh), "shear": shear}
    keep = np.isin(lab, [next(iter(g.labels)) for g in groups])
    ct, cb, xh = _core(keep)
    # 2. merge horizontally overlapping components
    groups.sort(key=lambda g: g.x0)
    changed = True
    while changed:
        changed, out = False, []
        for g in groups:
            if out:
                o = out[-1]
                ov = min(o.x1, g.x1) - max(o.x0, g.x0)
                if ov > 0 and ov >= p.overlap_merge * min(o.w, g.w):
                    o.merge(g); changed = True; continue
            out.append(g)
        groups = sorted(out, key=lambda g: g.x0)
    # 4. punctuation
    whole = _single_mark(groups, H, W)
    if whole is not None:
        return [{"x0": 0, "x1": W, "y0": 0, "y1": H, "kind": "punct", "char": whole, "mask": keep}], \
               {"core": (ct, cb, xh), "shear": shear}
    if len(groups) > 1:
        for g in groups:
            c = _punct(g, ct, cb, xh)
            if c is not None:
                g.kind, g.char = "punct", c
        letters = [g for g in groups if g.kind == "char"]
        if not letters:                       # everything looked like punctuation: treat as letters
            for g in groups:
                g.kind, g.char = "char", None
        # two apostrophes side by side -> one double quote
        merged = []
        for g in groups:
            if merged and g.char == "'" and merged[-1].char == "'" and g.x0 - merged[-1].x1 < 0.5 * xh:
                merged[-1].merge(g); merged[-1].char = '"'; continue
            merged.append(g)
        groups = merged
    # 5. split wide letter groups at projection-profile minima
    cw = max(p.width_factor * _stroke_width(keep), 3)
    pieces = []
    for g in groups:
        gm = np.isin(lab[:, g.x0:g.x1], list(g.labels))
        if g.kind != "char":
            pieces.append({"x0": g.x0, "x1": g.x1, "kind": "punct", "char": g.char, "gm": gm, "off": g.x0})
            continue
        k = int(round(g.w / cw))
        cuts = []
        if k >= 2 and g.w > p.split_ratio * cw:
            prof = gm.sum(0).astype(float)
            step = g.w / k
            for j in range(1, k):
                t = j * step
                lo, hi = int(max(1, t - 0.4 * step)), int(min(g.w - 1, t + 0.4 * step))
                if hi <= lo:
                    continue
                cols = np.arange(lo, hi)
                cost = prof[lo:hi] + 0.15 * prof.max() * np.abs(cols - t) / step
                cuts.append(int(cols[np.argmin(cost)]))
        edges = [0] + sorted(set(cuts)) + [g.w]
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                pieces.append({"x0": g.x0 + a, "x1": g.x0 + b, "kind": "char", "char": None,
                               "gm": gm[:, a:b], "off": g.x0 + a})
    # 6. merge thin letter pieces into the narrower neighbour
    def width(q): return q["x1"] - q["x0"]
    i = 0
    while i < len(pieces):
        q = pieces[i]
        if q["kind"] == "char" and width(q) < p.min_piece * cw and len(pieces) > 1:
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(pieces) and pieces[j]["kind"] == "char"]
            if nb:
                j = min(nb, key=lambda j: width(pieces[j]))
                a, b = sorted((i, j))
                L, R = pieces[a], pieces[b]
                x0, x1 = min(L["x0"], R["x0"]), max(L["x1"], R["x1"])
                gm = np.zeros((H, x1 - x0), bool)
                for s in (L, R):
                    gm[:, s["x0"] - x0:s["x1"] - x0] |= s["gm"]
                pieces[a] = {"x0": x0, "x1": x1, "kind": "char", "char": None, "gm": gm, "off": x0}
                del pieces[b]
                i = max(a - 1, 0)
                continue
        i += 1
    out = []
    for q in pieces:
        rows = np.nonzero(q["gm"].any(1))[0]
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        out.append({"x0": int(q["x0"]), "x1": int(q["x1"]), "y0": y0, "y1": y1, "kind": q["kind"],
                    "char": q["char"], "mask": q["gm"][y0:y1]})
    return out, {"core": (ct, cb, xh), "shear": shear}


def _stroke_width(m):
    """Mean stroke thickness in px (2 x area / perimeter)."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2.0 * m.sum() / max(per, 1.0)


def to_emnist(mask, size=28, box=24, target_stroke=2.8):
    """One character's ink mask -> 28x28 uint8 like EMNIST (white on black, long side 24 px,
    bounding box centered, stroke thickness matched to EMNIST, light blur)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((size, size), np.uint8)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = m.shape
    scale = box / max(h, w)
    sw = _stroke_width(m) * scale            # stroke width it would have at 28x28
    pad = 4
    m = np.pad(m, pad)
    if sw < 0.8 * target_stroke:
        k = int(np.clip(round((target_stroke - sw) / scale), 1, 15))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1)))
    elif sw > 1.6 * target_stroke:
        k = int(np.clip(round((sw - target_stroke) / scale / 2), 1, 15))
        er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1)))
        m = er if er.any() else m
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    scale = box / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    r = cv2.resize(m.astype(np.float32) * 255, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = r
    out = cv2.GaussianBlur(out, (3, 3), 0.7)
    return np.clip(out * (255.0 / max(out.max(), 1)), 0, 255).astype(np.uint8)


def deslanted_gray(gray, shear):
    """The grayscale word with the same shear as the mask (for figures)."""
    return _shear(gray, shear, 255) if shear else gray


def segment_word(gray, p: SegParams = SegParams(), return_info=False):
    """-> (pieces, crops) where crops is (n_char_pieces, 28, 28) uint8 for the 'char' pieces
    (piece coordinates are in the slant-corrected image; see deslanted_gray)."""
    pieces, info = segment(gray, p)
    crops = [to_emnist(q["mask"]) for q in pieces if q["kind"] == "char"]
    crops = np.stack(crops) if crops else np.zeros((0, 28, 28), np.uint8)
    return (pieces, crops, info) if return_info else (pieces, crops)


# ------------------------------------------------------------------ decoding with a character prior

def char_prior(train_texts, emnist_train_labels, power=0.3):
    """log p_IAM(c) - log p_EMNIST_eff(c) for the 62 EMNIST classes. p_IAM comes from the IAM
    *training* labels only; p_EMNIST_eff is the class frequency the classifier effectively saw
    (counts^(1-power) because of the class weights used in training)."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    iam = np.ones(len(CLASSES))
    for t in train_texts:
        for c in t:
            if c in idx:
                iam[idx[c]] += 1
    em = (np.bincount(emnist_train_labels, minlength=len(CLASSES)).astype(float) + 1) ** (1 - power)
    return np.log(iam / iam.sum()) - np.log(em / em.sum())


def decode(pieces, logp, prior=None, alpha=0.0):
    """Join the word: punctuation pieces keep their rule label, letter pieces take the best class
    of log p(class | crop) + alpha * prior."""
    s = logp if (prior is None or alpha == 0) else logp + alpha * prior[None, :]
    best = s.argmax(1) if len(s) else []
    out, k = [], 0
    for q in pieces:
        if q["kind"] == "punct":
            out.append(q["char"])
        else:
            out.append(CLASSES[int(best[k])]); k += 1
    return "".join(out)


def params_dict(p):
    return asdict(p)

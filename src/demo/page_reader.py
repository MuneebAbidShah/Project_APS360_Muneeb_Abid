"""PAGE mode (beta) of the live demo: a running transcript of a whole page.

WORD mode (one word in the guide box) is the reliable mode; page mode adds three error sources
(finding the paper, cutting it into words, tracking words between frames), so it is less accurate.

Pipeline, about twice a second:
  1. PageSegmenter: find the paper (largest bright 4-corner shape) and straighten it with a
     perspective warp (if no paper outline is found, the whole frame is the page); flatten the
     lighting and binarize (webcam.flatten_lighting + Otsu); remove ruled lines; split into
     lines with the horizontal projection (one peak per line, cut at the emptiest row between
     peaks); connected components are given to the line their centre falls in, then joined into
     words left to right when the gap is below `word_gap` x the typical letter height;
     heuristics drop specks, drawings and page edges.
  2. Each word crop is read by the CRNN with exactly the word-mode preprocessing
     (webcam.clean_photo -> training transform -> greedy CTC).
  3. WordTracker de-duplicates by POSITION, never by text: boxes live in paper coordinates
     (straightened page; remaining drift is removed by registering each frame's boxes to the
     tracked boxes with the median offset, plus an IoU/overlap tolerance);
     a box is committed only after it has been stable for `commit_after` updates with a
     consistent majority-vote reading; a committed box is never added twice (its text may be
     refined); ink growing at the right edge of a word updates that word instead of adding one.
  4. Transcript: committed words ordered by line, then by x.
"""
import collections
from dataclasses import dataclass, field

import numpy as np

from src.demo.webcam import clean_photo, flatten_lighting


def _cv2():
    import cv2
    return cv2


# ------------------------------------------------------------------ geometry helpers

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_small(a, b):
    """Intersection / area of the smaller box (1.0 when one box contains the other)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def shift_box(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


# ------------------------------------------------------------------ 1. segmentation

@dataclass
class SegResult:
    page: np.ndarray                 # straightened, lighting-flattened gray page (uint8)
    color: np.ndarray                # straightened colour page (for saving crops)
    boxes: list                      # word boxes (x0, y0, x1, y1) in page coordinates
    lines: list                      # line index for every box
    to_frame: object = None          # 3x3 matrix page -> frame (None = identity)
    text_height: float = 0.0
    paper_found: bool = False


class PageSegmenter:
    def __init__(self, page_width=1000, straighten=True, word_gap=0.45, min_paper_frac=0.2):
        self.page_width, self.straighten = page_width, straighten
        self.word_gap = word_gap                 # letters closer than word_gap x text height are one word
        self.min_paper_frac = min_paper_frac

    # -- paper
    def find_paper(self, gray):
        """4 corners (tl, tr, br, bl) of the paper, or None."""
        cv2 = _cv2()
        h, w = gray.shape
        s = 400 / max(h, w)
        small = cv2.GaussianBlur(cv2.resize(gray, (int(w * s), int(h * s))), (5, 5), 0)
        _, bright = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cs, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_paper_frac * small.size or area > 0.97 * small.size:
            return None                           # no outline (paper fills the frame) or too small
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None
        p = approx.reshape(4, 2).astype(np.float32) / s
        ssum, diff = p.sum(1), np.diff(p, axis=1).ravel()
        return np.float32([p[ssum.argmin()], p[diff.argmin()], p[ssum.argmax()], p[diff.argmax()]])

    def warp(self, frame, gray):
        cv2 = _cv2()
        corners = self.find_paper(gray) if self.straighten else None
        if corners is None:
            return frame, gray, None, False
        tl, tr, br, bl = corners
        pw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        ph = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        W = self.page_width; H = int(W * ph / max(pw, 1))
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
        M = cv2.getPerspectiveTransform(corners, dst)
        inset = lambda im: cv2.warpPerspective(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        return inset(frame), inset(gray), np.linalg.inv(M), True

    # -- ink and words
    def ink(self, page_gray):
        cv2 = _cv2()
        flat = flatten_lighting(page_gray)
        u8 = flat.astype(np.uint8)
        t, _ = cv2.threshold(cv2.GaussianBlur(u8, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = min(t, 200)                           # an almost-empty page: Otsu would pick paper noise
        m = (u8 < t).astype(np.uint8)
        h, w = m.shape
        lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 6), 1)))
        m[lines > 0] = 0                          # ruled lines / page edges
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        e = 2 + int(0.008 * min(h, w))            # page border: leftovers of the desk / paper edge
        m[:e] = 0; m[-e:] = 0; m[:, :e] = 0; m[:, -e:] = 0
        return flat.astype(np.uint8), m

    def segment(self, frame):
        cv2 = _cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        color, pgray, to_frame, found = self.warp(frame, gray)
        flat, m = self.ink(pgray)
        H, W = m.shape
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        comps = [(int(x), int(y), int(x + w), int(y + h), int(a), float(cy))
                 for (x, y, w, h, a), (_, cy) in zip(st[1:], cen[1:]) if a >= 4]
        big = [c for c in comps if c[4] >= max(12, 2e-5 * m.size)]
        if not big:
            return SegResult(flat, color, [], [], to_frame, 0.0, found)
        areas = np.array([c[4] for c in big])
        th = float(np.median([c[3] - c[1] for c in big if c[4] >= np.percentile(areas, 30)]))
        comps = [c for c in comps if (c[3] - c[1]) < 5 * th and (c[2] - c[0]) < 0.95 * W]   # drop drawings/edges
        bands = self._line_bands(m, th)
        rows = [[] for _ in bands]
        for c in comps:
            li = int(np.searchsorted([b1 for _, b1 in bands], c[5]))
            if li < len(bands):
                rows[li].append(c)
        boxes, lines = [], []
        k = self.word_gap * th
        for li, row in enumerate(rows):
            words = []
            for c in sorted(row):
                if words and c[0] - words[-1][2] < k:          # small gap (or overlap): same word
                    w = words[-1]
                    words[-1] = [min(w[0], c[0]), min(w[1], c[1]), max(w[2], c[2]), max(w[3], c[3]), w[4] + c[4]]
                else:
                    words.append(list(c[:5]))
            for w in words:
                if w[4] >= max(12, 0.05 * th * th) and (w[3] - w[1]) >= 0.25 * th:   # not just a speck
                    boxes.append(tuple(w[:4])); lines.append(li)
        # renumber lines that ended up empty
        used = sorted(set(lines)); ren = {l: i for i, l in enumerate(used)}
        lines = [ren[l] for l in lines]
        return SegResult(flat, color, boxes, lines, to_frame, th, found)

    def _line_bands(self, m, th):
        """[(y_start, y_end)] per text line from the horizontal projection: smoothed ink per row,
        one peak per line (peaks at least ~0.9 text heights apart), cut at the emptiest row between
        two peaks. Touching ascenders/descenders are fine: a shallow valley still separates lines."""
        cv2 = _cv2()
        prof = m.sum(1).astype(np.float32)
        sig = max(1.0, 0.3 * th)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), sig).ravel() if len(prof) > 3 else prof
        if prof.max() <= 0:
            return []
        cand = [y for y in range(1, len(prof) - 1) if prof[y] >= prof[y - 1] and prof[y] >= prof[y + 1]
                and prof[y] > 0.12 * prof.max()]
        peaks = []
        for y in sorted(cand, key=lambda y: -prof[y]):            # strongest first, keep them apart
            if all(abs(y - q) > 0.9 * th for q in peaks):
                peaks.append(y)
        peaks.sort()
        merged = [peaks[0]]
        for q in peaks[1:]:                                       # no real valley in between: same line
            v = prof[merged[-1]:q + 1].min()
            if v > 0.75 * min(prof[merged[-1]], prof[q]):
                merged[-1] = merged[-1] if prof[merged[-1]] >= prof[q] else q
            else:
                merged.append(q)
        cuts = [0] + [int(p + np.argmin(prof[p:q + 1])) for p, q in zip(merged, merged[1:])] + [len(prof)]
        return list(zip(cuts[:-1], cuts[1:]))


# ------------------------------------------------------------------ 3. tracking / de-duplication

@dataclass
class Track:
    id: int
    box: tuple
    votes: collections.deque = field(default_factory=lambda: collections.deque(maxlen=7))
    stable: int = 1
    missed: int = 0
    committed: bool = False
    text: str = ""
    age: int = 0
    last_read: int = -99

    def majority(self):
        if not self.votes:
            return "", 0.0
        t, n = collections.Counter(self.votes).most_common(1)[0]
        return t, n / len(self.votes)


class WordTracker:
    """Matches word boxes between frames by position and commits each word exactly once."""

    def __init__(self, commit_after=3, agree=0.6, iou_same=0.55, forget_pending=4):
        self.commit_after, self.agree, self.iou_same, self.forget = commit_after, agree, iou_same, forget_pending
        self.tracks, self.next_id, self.step = [], 1, 0

    def reset(self):
        self.tracks, self.next_id, self.step = [], 1, 0

    def match(self, boxes):
        """For each box: the index of the track it belongs to, or None (new word), or -1 (ignore:
        the box covers two or more committed words, e.g. two words merged by the splitter)."""
        out, used = [], set()
        for b in boxes:
            cands = [(iou(b, t.box), overlap_small(b, t.box), i) for i, t in enumerate(self.tracks)]
            covering = [i for s, o, i in cands if o > 0.7 and self.tracks[i].committed]
            if len(covering) >= 2:
                out.append(-1); continue
            good = [(s, o, i) for s, o, i in cands if (s > 0.3 or o > 0.7) and i not in used]
            if not good:
                out.append(None); continue
            s, o, i = max(good)
            used.add(i); out.append(i)
        return out

    def needs_read(self, idx, box):
        """Committed words whose box hasn't changed are re-read only every 4th update (CPU time)."""
        if idx is None or idx < 0:
            return idx is None
        t = self.tracks[idx]
        return not (t.committed and iou(box, t.box) > 0.85 and self.step - t.last_read < 4)

    def update(self, boxes, texts, matches=None):
        """boxes: detections in reference-paper coordinates; texts: CRNN reading per box (None = not
        re-read this time). Returns the ids of newly committed tracks."""
        self.step += 1
        matches = self.match(boxes) if matches is None else matches
        seen, new_commits = set(), []
        for b, txt, i in zip(boxes, texts, matches):
            if i == -1:
                continue
            if i is None:
                t = Track(self.next_id, tuple(b)); self.next_id += 1
                if txt is not None:
                    t.votes.append(txt); t.last_read = self.step
                self.tracks.append(t); seen.add(t.id)
                continue
            t = self.tracks[i]; seen.add(t.id)
            same = iou(b, t.box) > self.iou_same
            h = t.box[3] - t.box[1]
            grew_right = ((b[2] - t.box[2]) > 0.15 * h and abs(b[0] - t.box[0]) < 0.4 * h
                          and overlap_small(b, t.box) > 0.85)          # the old box is (almost) inside the new one
            if grew_right:                        # the word is still being written: same word, new box
                t.box, t.stable = tuple(b), 1
                t.votes.clear()
            elif same:
                t.stable += 1
                t.box = tuple(int(round(0.7 * u + 0.3 * v)) for u, v in zip(t.box, b))   # smooth jitter
            elif not t.committed:                 # a different box (e.g. a bad cut last time): start over
                t.box, t.stable = tuple(b), 1
                t.votes.clear()
            else:                                 # committed word seen with a different box: keep it as is
                t.stable = 1
            if txt is not None:
                t.votes.append(txt); t.last_read = self.step
            t.missed = 0
            text, frac = t.majority()
            if not t.committed and t.stable >= self.commit_after and text and frac >= self.agree:
                t.committed, t.text = True, text
                new_commits.append(t.id)
            elif t.committed and text and frac >= self.agree and same:
                t.text = text                     # refine, never duplicate
        keep = []
        for t in self.tracks:
            if t.id not in seen:
                t.missed += 1
            t.age += 1
            if t.committed or t.missed <= self.forget:
                keep.append(t)                    # committed words stay even when out of view
        self.tracks = keep
        return new_commits

    @property
    def committed(self):
        return [t for t in self.tracks if t.committed]

    @property
    def pending(self):
        return [t for t in self.tracks if not t.committed]


class Transcript:
    """Committed words ordered by line (vertical centre), then by x."""

    @staticmethod
    def lines(tracks):
        ts = sorted(tracks, key=lambda t: (t.box[1] + t.box[3]) / 2)
        if not ts:
            return []
        h = float(np.median([t.box[3] - t.box[1] for t in ts]))
        rows, centers = [], []
        for t in ts:
            c = (t.box[1] + t.box[3]) / 2
            if centers and abs(c - np.mean(centers[-1])) < 0.5 * h:
                rows[-1].append(t); centers[-1].append(c)
            else:
                rows.append([t]); centers.append([c])
        return [sorted(r, key=lambda t: t.box[0]) for r in rows]

    @classmethod
    def text(cls, tracks):
        return "\n".join(" ".join(t.text for t in row) for row in cls.lines(tracks))


# ------------------------------------------------------------------ everything together

class PageReader:
    def __init__(self, recognizer, segmenter=None, tracker=None, align=True):
        self.rec = recognizer
        self.seg = segmenter or PageSegmenter()
        self.tracker = tracker or WordTracker()
        self.align = align
        self.shift = (0.0, 0.0)                    # page coords -> reference (track) coords
        self.last = None                           # last SegResult
        self.last_crops = []                       # (box in page coords, colour crop, reading) of the last frame
        self.last_model_input = None

    def reset(self):
        self.tracker.reset(); self.shift = (0.0, 0.0)

    def _register(self, boxes):
        """Shift (dx, dy) from this frame's page coordinates to the reference (track) coordinates:
        start from the last shift, pair every detection with the track it overlaps most, and move by
        the median offset of those pairs (twice). Robust to new words appearing (unlike image
        correlation) and to a few wrong pairs; follows the paper drifting a little every frame."""
        dx, dy = self.shift
        tracks = [t for t in self.tracker.tracks if t.committed or t.stable >= 2]
        if not tracks or not boxes:
            return dx, dy
        c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for _ in range(2):
            offs = []
            for b in boxes:
                sb = shift_box(b, dx, dy)
                best = max(tracks, key=lambda t: iou(sb, t.box))
                if iou(sb, best.box) > 0.3:
                    (tx, ty), (bx, by) = c(best.box), c(sb)
                    offs.append((tx - bx, ty - by))
            if not offs:
                break
            mx, my = np.median(offs, axis=0)
            dx, dy = dx + mx, dy + my
        return dx, dy

    def process(self, frame, force_read=False):
        """Read one frame. force_read=True re-reads every word (page capture)."""
        from src.demo.webcam import tensor_to_u8
        r = self.seg.segment(frame)
        self.last = r
        self.shift = self._register(r.boxes) if self.align else (0.0, 0.0)
        dx, dy = self.shift
        ref_boxes = [tuple(int(round(v)) for v in shift_box(b, dx, dy)) for b in r.boxes]
        matches = self.tracker.match(ref_boxes)
        texts, crops = [], []
        for b, rb, mi in zip(r.boxes, ref_boxes, matches):
            pad = int(0.15 * (b[3] - b[1])) + 2
            y0, y1 = max(0, b[1] - pad), min(r.page.shape[0], b[3] + pad)
            x0, x1 = max(0, b[0] - pad), min(r.page.shape[1], b[2] + pad)
            txt = None
            if force_read or self.tracker.needs_read(mi, rb):
                g = clean_photo(r.page[y0:y1, x0:x1])
                if g is not None:
                    x = self.rec.tensor(g)
                    txt = self.rec.predict_tensor(x)
                    self.last_model_input = tensor_to_u8(x)
                else:
                    txt = ""
            texts.append(txt)
            crops.append(((x0, y0, x1, y1), r.color[y0:y1, x0:x1].copy(), txt))
        self.last_crops = crops
        self.tracker.update(ref_boxes, texts, matches)
        return r

    def transcript(self):
        return Transcript.text(self.tracker.committed)

    def frame_box(self, ref_box):
        """A reference-paper box -> 4 frame points (for drawing)."""
        dx, dy = self.shift
        x0, y0, x1, y1 = shift_box(ref_box, -dx, -dy)
        pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        M = self.last.to_frame if self.last is not None else None
        if M is not None:
            ph = np.hstack([pts, np.ones((4, 1), np.float32)]) @ M.T
            pts = ph[:, :2] / ph[:, 2:3]
        return pts.astype(np.int32)

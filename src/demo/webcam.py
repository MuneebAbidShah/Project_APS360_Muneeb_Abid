"""Webcam / phone photo of a handwritten word -> the kind of image the CRNN was trained on.

Used by BOTH the live demo on the laptop (live_demo.py) and the Step 5 notebook that scores
the webcam test set, so the two always preprocess identically.

IAM word images (the training data) are tight crops of dark ink on near-white paper:
background ~250, darkest ink ~60, no margin around the ink. A photo of a word on paper is
different: grey, unevenly lit paper, a lot of empty paper around the word, sometimes ruled
lines, a shadow or the edge of the page. `clean_photo` closes that gap in five steps:

  1. grayscale;
  2. flatten the lighting: estimate the paper brightness everywhere with a max-filter that is
     wider than a pen stroke (so it removes the ink), and divide by it -> paper becomes white;
  3. find the ink with Otsu's threshold, then drop specks (tiny components), thin lines that run
     across the whole box (ruled paper) and big dark blobs coming in from the edge (shadows);
  4. crop tightly to the remaining ink (IAM crops have no margin);
  5. set the paper to IAM's background level (250); if the ink is fainter (or darker) than
     anything common in IAM, stretch it so its darkest 10% falls in IAM's range (33-110), and
     blank the paper away from the ink. Pen strokes far thinner or thicker than IAM's (relative
     to the word height) are thickened/thinned by one step.

Checked on 720 real IAM test words (already clean, so the clean-up should change little):
CRNN CER with clean_photo is within ~0.5 points of the CER without it.

The result then goes through **exactly the training transform** (src/data/transforms.py:
resize to 32 px high keeping the aspect ratio, clamp the width to 16-512 px, scale to [-1, 1]),
and the CRNN's greedy CTC decoding (src/data/charset.py). An IAM image passed with
clean=False gives exactly the tensor the evaluation used.
"""
import numpy as np
import torch
from PIL import Image

from src.config import ModelConfig
from src.data.charset import Charset
from src.data.transforms import Preprocess
from src.models.crnn import CRNN

# Measured on 400 IAM word images (median): paper 250, darkest 10% of the ink 58,
# stroke width / word height 0.049 (10th-90th percentile 0.027-0.106).
IAM_PAPER, IAM_INK = 250, 58
IAM_INK_RANGE = (33, 110)     # 5th-95th percentile was 33-125
STROKE_RANGE = (0.025, 0.11)
MODEL_FORMAT = "aps360-crnn-cpu-v1"


def _cv2():
    import cv2          # imported lazily so the model code works without OpenCV
    return cv2


def to_gray(img):
    """BGR/RGB/gray uint8 array or PIL image -> 2-D uint8 gray array."""
    cv2 = _cv2()
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"))
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.uint8)


def flatten_lighting(gray):
    """Divide by the local paper brightness so uneven light/shadows become white paper."""
    cv2 = _cv2()
    h, w = gray.shape
    k = max(15, (min(h, w) // 3) | 1)                   # much wider than a pen stroke
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4).astype(np.float32)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255)


def ink_mask(flat, min_contrast=25):
    """Boolean ink mask after cleanup, or None if there is no clear writing in the box."""
    cv2 = _cv2()
    u8 = flat.astype(np.uint8)
    t, _ = cv2.threshold(cv2.GaussianBlur(u8, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (u8 < t).astype(np.uint8)
    if not m.any() or (255 - np.percentile(flat[m > 0], 10)) < min_contrast:
        return None                                      # nothing darker than the paper: empty box
    h, w = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(st[1:, 4]))
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        ruled_line = x == 0 and x + bw == w and bh < 0.12 * h               # thin line across the whole box
        speck = area < max(4, 0.00015 * h * w) and area < 0.03 * st[largest, 4]   # noise dot (i-dots are kept)
        keep[i] = i == largest or not (ruled_line or speck)
    m = keep[lab]
    if m.sum() < 4:
        return None
    return m


def _stroke_ratio(m):
    cv2 = _cv2()
    if m.all():                       # no background pixel at all: the distance would be "infinite"
        return 1.0
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    ys = np.nonzero(m.any(1))[0]
    return 2 * float(d[m].mean()) / max(1, ys[-1] - ys[0] + 1)


def clean_photo(img, adjust_stroke=True, mask_grow=3):
    """Photo crop of one word -> IAM-like uint8 gray image (white paper, dark ink, tight crop).
    Returns None when the box has no clear writing."""
    cv2 = _cv2()
    gray = to_gray(img)
    if gray.size == 0 or min(gray.shape) < 3:
        return None
    flat = flatten_lighting(gray)
    m = ink_mask(flat)
    if m is None:
        return None
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    flat, m = flat[y0:y1, x0:x1], m[y0:y1, x0:x1]
    if adjust_stroke and m.sum() > 20:
        r = _stroke_ratio(m)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if r < STROKE_RANGE[0]:                          # very thin pen for the word size: thicken
            m2 = cv2.dilate(m.astype(np.uint8), k).astype(bool)
            flat = np.where(m2 & ~m, np.minimum(flat, np.percentile(flat[m], 50)), flat); m = m2
        elif r > STROKE_RANGE[1]:                        # marker-thick strokes: thin a little
            m2 = cv2.erode(m.astype(np.uint8), k).astype(bool)
            if m2.sum() > 20:
                flat = np.where(m & ~m2, np.maximum(flat, np.percentile(flat[m], 75)), flat); m = m2
    # Contrast: only pull the ink into IAM's range (darkest 10% between 33 and 110); ink that is
    # already IAM-like is left alone (forcing every word to one darkness level hurt on IAM itself).
    dark = float(np.percentile(flat[m], 10))
    target = min(max(dark, IAM_INK_RANGE[0]), IAM_INK_RANGE[1])
    out = IAM_PAPER - (255 - flat) * (IAM_PAPER - target) / max(1.0, 255 - dark)
    if mask_grow:                                        # paper away from the ink -> plain paper (no noise/texture)
        near = cv2.dilate(m.astype(np.uint8), np.ones((mask_grow, mask_grow), np.uint8)).astype(bool)
        out = np.where(near, out, IAM_PAPER)
    return np.clip(out, 0, 255).astype(np.uint8)


def guide_box(shape, wfrac=0.7, hfrac=0.25):
    """(x0, y0, x1, y1) of the centred box the word should be written in (frame shape h, w)."""
    h, w = shape[:2]
    bw, bh = int(w * wfrac), int(h * hfrac)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return x0, y0, x0 + bw, y0 + bh


def crop(frame, box):
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1].copy()


# ------------------------------------------------------------------ the model on the CPU

def export_model(checkpoint, out_path, extra=None):
    """Training checkpoint (best.pt, ~60 MB with the optimizer) -> small CPU file with only
    float32 weights + the model settings + the charset + the preprocessing settings."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from src.config import DataConfig
    d = DataConfig()
    out = {"format": MODEL_FORMAT,
           "model": {k: v.float().cpu() if v.is_floating_point() else v.cpu() for k, v in ck["model"].items()},
           "model_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(ck["model_cfg"]).items()},
           "charset": ck["charset"],
           "preprocess": {"height": d.img_height, "min_width": d.min_width, "max_width": d.max_width},
           "info": {"epoch": int(ck.get("epoch", -1)), "best_val_cer": float(ck.get("best_cer", float("nan"))),
                    **(extra or {})}}
    torch.save(out, out_path)
    return out


class Recognizer:
    """The CRNN on the CPU. Loads the exported file (or a training best.pt)."""

    def __init__(self, model_path, device="cpu", threads=None):
        if threads:
            torch.set_num_threads(threads)
        try:
            ck = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:                                # a training checkpoint (has optimizer state etc.)
            ck = torch.load(model_path, map_location=device, weights_only=False)
        mc = dict(ck["model_cfg"]); mc["cnn_channels"] = tuple(mc["cnn_channels"])
        self.model_cfg = dict(mc)                          # kept so two models can be checked for the same architecture
        self.charset = Charset(ck["charset"])
        self.model = CRNN(self.charset.num_classes, ModelConfig(**mc)).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.pre = Preprocess(**ck.get("preprocess", {}))   # the training transform (no augmentation)
        self.preprocess_cfg = {"height": self.pre.height, "min_width": self.pre.min_width,
                               "max_width": self.pre.max_width}          # effective values (defaults filled in)
        self.device = torch.device(device)
        self.info = ck.get("info", {"epoch": ck.get("epoch")})

    def tensor(self, gray_u8):
        """uint8 gray word -> (1, 32, W) tensor in [-1, 1], exactly as in training/evaluation."""
        return self.pre(Image.fromarray(gray_u8))

    @torch.no_grad()
    def predict_tensor(self, x):
        lp = self.model(x.unsqueeze(0).to(self.device))          # (T, 1, C)
        return self.charset.ctc_greedy_decode(lp[:, 0].argmax(1).tolist())

    def read(self, img, clean=True):
        """-> (text, model input as uint8 32-px image, cleaned word or None).
        clean=False: `img` is already an IAM-style word image (skips the photo clean-up)."""
        g = clean_photo(img) if clean else to_gray(img)
        if g is None:
            return "", None, None
        x = self.tensor(g)
        return self.predict_tensor(x), tensor_to_u8(x), g


def tensor_to_u8(x):
    """(1, H, W) tensor in [-1, 1] -> uint8 image (to show what the model sees)."""
    return ((x[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

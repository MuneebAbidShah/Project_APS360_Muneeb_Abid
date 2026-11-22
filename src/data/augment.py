"""Camera-style data augmentation for Model B (training split ONLY).

Model A (Step 2) only ever saw clean scans. A phone or webcam photo of handwriting looks
different: the page is tilted or seen at an angle, the hand shakes (motion blur), lighting is
uneven or a shadow falls across the word, contrast is lower, and the picture is noisy and
JPEG-compressed. Augmentation shows the model random versions of these "camera problems" on the
IAM training words, so it learns to read through them. The label never changes.

Rules this file enforces:
- only the TRAINING split is wrapped (AugmentedWords); validation and test stay clean,
  so Model A and Model B are compared on exactly the same clean IAM words;
- every distortion keeps the image's size and dtype (uint8 grayscale in, same shape out);
- it is reproducible: the random choices for word i in epoch e come from (seed, epoch, i), so
  the same seed gives the same images (also with several DataLoader workers);
- with train=False the wrapper returns the clean (un-augmented) image.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import Preprocess, to_tensor_normalized

cv2.setNumThreads(0)          # DataLoader workers do the parallel work; avoid oversubscription


@dataclass
class AugConfig:
    """Probability and strength of each distortion (one place to change them)."""
    p_perspective: float = 0.5    # seen at an angle: corners moved by up to `perspective` of h
    perspective: float = 0.12
    p_rotate: float = 0.5         # tilt of the paper / camera
    max_rotate_deg: float = 5.0
    p_shear: float = 0.3          # slant change
    max_shear: float = 0.25
    p_motion_blur: float = 0.35   # shaky hand / moving camera
    max_blur_len: int = 7         # pixels at the 64-px working height
    p_gauss_blur: float = 0.2     # out of focus
    p_lighting: float = 0.5       # smooth uneven lighting (brighter on one side)
    p_shadow: float = 0.3         # a soft dark band across the word
    p_contrast: float = 0.5       # faded ink / grey paper
    p_noise: float = 0.3          # sensor noise
    max_noise_std: float = 10.0
    p_jpeg: float = 0.3           # phone / webcam compression
    jpeg_quality: tuple = (25, 80)

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def _paper(img):
    """Grey level of the paper (bright background), used to fill borders after a warp."""
    return int(np.percentile(img, 90))


def perspective_warp(img, rng, amount):
    h, w = img.shape
    d = amount * h
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.uniform(-d, d, size=(4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=_paper(img))


def rotate(img, rng, max_deg):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-max_deg, max_deg), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_paper(img))


def shear(img, rng, max_shear):
    h, w = img.shape
    s = rng.uniform(-max_shear, max_shear)
    M = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_paper(img))


def motion_blur(img, rng, max_len):
    n = int(rng.integers(3, max(4, max_len + 1)))
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, M, (n, n))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)


def gaussian_blur(img, rng):
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.4))


def uneven_lighting(img, rng):
    """Multiply by a smooth brightness ramp in a random direction (0.55-1.0 of the brightness)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    t = (np.cos(a) * xx / max(w, 1) + np.sin(a) * yy / max(h, 1))
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    lo = rng.uniform(0.55, 0.9)
    return img.astype(np.float32) * (lo + (1 - lo) * t)


def shadow(img, rng):
    """A soft dark band (like the shadow of a hand or phone) across part of the word."""
    h, w = img.shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    a = rng.uniform(-0.6, 0.6)                       # band direction (mostly vertical)
    c = rng.uniform(0, w)
    width = rng.uniform(0.15, 0.5) * w
    dist = np.abs(xx - c - a * (yy - h / 2))
    band = np.clip(1 - dist / width, 0, 1) ** 2
    return img.astype(np.float32) * (1 - rng.uniform(0.25, 0.55) * band)


def contrast(img, rng):
    """Lower contrast (faded ink, grey paper) and shift brightness a little."""
    f = img.astype(np.float32)
    m = f.mean()
    return (f - m) * rng.uniform(0.45, 1.0) + m + rng.uniform(-25, 15)


def noise(img, rng, max_std):
    return img.astype(np.float32) + rng.normal(0, rng.uniform(2, max_std), img.shape).astype(np.float32)


def jpeg(img, rng, quality):
    q = int(rng.integers(quality[0], quality[1] + 1))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else None
    return out if out is not None and out.shape == img.shape else img


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class CameraAugment:
    """img (uint8 HxW grayscale) + numpy Generator -> distorted uint8 image of the same shape.
    Geometry first (as the camera sees the page), then optics (blur), then light, then the sensor
    (noise) and finally compression, which is the order they happen in a real photo."""

    def __init__(self, cfg: AugConfig | None = None):
        self.cfg = cfg or AugConfig()

    def __call__(self, img, rng):
        c = self.cfg
        assert img.dtype == np.uint8 and img.ndim == 2, "expects a uint8 grayscale image"
        if min(img.shape) < 4:
            return img.copy()
        x = img
        if rng.random() < c.p_perspective:
            x = perspective_warp(x, rng, c.perspective)
        if rng.random() < c.p_rotate:
            x = rotate(x, rng, c.max_rotate_deg)
        if rng.random() < c.p_shear:
            x = shear(x, rng, c.max_shear)
        if rng.random() < c.p_motion_blur:
            x = motion_blur(x, rng, c.max_blur_len)
        elif rng.random() < c.p_gauss_blur:
            x = gaussian_blur(x, rng)
        if rng.random() < c.p_lighting:
            x = _u8(uneven_lighting(x, rng))
        if rng.random() < c.p_shadow:
            x = _u8(shadow(x, rng))
        if rng.random() < c.p_contrast:
            x = _u8(contrast(x, rng))
        if rng.random() < c.p_noise:
            x = _u8(noise(x, rng, c.max_noise_std))
        if rng.random() < c.p_jpeg:
            x = jpeg(_u8(x), rng, c.jpeg_quality)
        x = _u8(x)
        assert x.shape == img.shape
        return x


class AugmentedWords(Dataset):
    """Wraps the IAM TRAINING dataset for Model B.

    The training images are kept in RAM at `work_height` px (64, twice the model's 32), so the
    distortions are applied at a higher resolution than the model sees; each sample is then
    resized to 32 px and normalised with the SAME Preprocess as training/eval (src/data/transforms.py).
    Same items, labels, skip list and bad-image handling as the wrapped dataset: images that can't be
    decoded are dropped and listed in .dropped (counted by add_dropped()).
    """

    def __init__(self, base, augment=None, seed=0, work_height=64, train=True, num_threads=8, preload=True):
        self.base, self.items, self.charset = base, list(base.items), base.charset
        self.transform = base.transform                      # 32-px resize + normalise (unchanged)
        self.work = Preprocess(height=work_height, min_width=base.transform.min_width * work_height // base.transform.height,
                               max_width=base.transform.max_width * work_height // base.transform.height)
        self.augment = augment or CameraAugment()
        self.seed, self.train = seed, train
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()   # seen by DataLoader workers
        self.dropped = list(getattr(base, "dropped", []))
        self._cache = None
        if preload:
            self.preload(num_threads)

    def set_epoch(self, epoch: int):
        self._epoch[0] = int(epoch)

    @property
    def epoch(self):
        return int(self._epoch[0])

    def _load(self, it):
        try:
            with Image.open(it.path) as im:
                img = im.convert("L")
            if img.size[0] == 0 or img.size[1] == 0:
                return None
            return np.asarray(self.work.resize(img), dtype=np.uint8)
        except Exception:
            return None

    def preload(self, num_threads=8):
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            cache = list(pool.map(self._load, self.items))
        bad = [i for i, a in enumerate(cache) if a is None]
        if bad:
            ids = [self.items[i].word_id for i in bad]
            print(f"dropped {len(ids)} unreadable image(s) while preloading: {', '.join(ids[:10])}")
            self.dropped += ids
            keep = set(range(len(cache))) - set(bad)
            self.items = [it for i, it in enumerate(self.items) if i in keep]
            cache = [a for i, a in enumerate(cache) if i in keep]
        self._cache = cache
        return self

    def __len__(self):
        return len(self.items)

    def widths(self):
        """Width of each sample after the final 32-px resize (for batching similar widths)."""
        return [self.transform.resized_width(a.shape[1], a.shape[0]) for a in self._cache]

    def rng(self, idx, epoch=None):
        return np.random.default_rng([int(self.seed), int(self.epoch if epoch is None else epoch), int(idx)])

    def image(self, idx, epoch=None, augment=None):
        """The uint8 working-resolution image the model input is made from (augmented if train)."""
        a = self._cache[idx]
        if augment if augment is not None else self.train:
            a = self.augment(a, self.rng(idx, epoch))
        return a

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.fromarray(self.image(idx))
        x = to_tensor_normalized(self.transform.resize(img))
        target = torch.tensor(self.charset.encode(it.text), dtype=torch.long)
        return x, target, len(target), it.text

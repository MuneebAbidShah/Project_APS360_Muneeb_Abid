"""EMNIST character classifier: data download, a small CNN, and resumable training.

Second baseline from the proposal: segment each IAM word into characters (see char_segment.py)
and classify every crop with a CNN trained only on EMNIST (no IAM training at all).

Split: **byclass** (62 classes: 0-9, A-Z, a-z). It is the only EMNIST split that keeps upper- and
lower-case letters apart, which our case-sensitive CER needs ('balanced' merges 15 lower-case letters
such as c/o/s into their capitals). byclass is unbalanced (many digits), so training uses mildly
re-weighted classes and recognition can use the IAM *training* character frequencies as a prior.

Data sources, tried in order, no sign-in needed (all serve the original NIST idx files):
  1. Hugging Face mirror  Royc30ne/emnist-byclass
  2. Hugging Face mirror  Heliosoph/EMNIST
  3. NIST original        biometrics.nist.gov gzip.zip (562 MB, all splits)
Each file is checked (gzip + idx header + item count) before it is used.

EMNIST images are stored **transposed** (a known quirk of the release). `orient()` checks the
mean image of the digit '1' (a vertical stroke when upright) and transposes if needed, so the
code is right whichever way a mirror stores them. Images are white ink on black.
"""
import gzip
import io
import json
import os
import shutil
import struct
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # byclass label order
FILES = {
    "train_x": "emnist-byclass-train-images-idx3-ubyte.gz",
    "train_y": "emnist-byclass-train-labels-idx1-ubyte.gz",
    "test_x": "emnist-byclass-test-images-idx3-ubyte.gz",
    "test_y": "emnist-byclass-test-labels-idx1-ubyte.gz",
}
COUNTS = {"train": 697_932, "test": 116_323}
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
SOURCES = [
    ("Hugging Face mirror Royc30ne/emnist-byclass", "hf", "Royc30ne/emnist-byclass"),
    ("Hugging Face mirror Heliosoph/EMNIST", "hf", "Heliosoph/EMNIST"),
    ("NIST original gzip.zip (562 MB)", "nist", "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"),
]


# ------------------------------------------------------------------ download + reading

def read_idx(path):
    """Read a (gzipped) idx file into a uint8 numpy array."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError(f"{path}: not a uint8 idx file")
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path}: truncated ({arr.size} of {int(np.prod(shape))} values)")
    return arr.reshape(shape)


def _valid(path, expect_n=None):
    try:
        with gzip.open(path, "rb") as f:
            head = f.read(8)
        zero, dtype, ndim, n = struct.unpack(">HBBI", head)
        return zero == 0 and dtype == 8 and (expect_n is None or n == expect_n)
    except Exception:
        return False


def _expected(key):
    return COUNTS["train" if key.startswith("train") else "test"]


def _fetch(url, dest, timeout=60, retries=3):
    tmp = Path(str(dest) + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "aps360-htr"}),
                                        timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"    attempt {attempt}/{retries} failed: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 * attempt)
    tmp.unlink(missing_ok=True)
    return False


def download(dest_dir, sources=SOURCES, verbose=True):
    """Make sure the four byclass files are in `dest_dir` (skips files already there and valid).
    Tries each source in turn; returns the name of the source used (or 'cached')."""
    dest = Path(dest_dir); dest.mkdir(parents=True, exist_ok=True)
    def missing():
        return [k for k, n in FILES.items() if not _valid(dest / n, _expected(k))]
    if not missing():
        return "cached"
    for label, kind, where in sources:
        if verbose:
            print(f"downloading EMNIST byclass from {label} ...")
        t = time.time()
        if kind == "hf":
            for k in missing():
                if not _fetch(HF.format(repo=where, name=FILES[k]), dest / FILES[k]) or not _valid(dest / FILES[k], _expected(k)):
                    print(f"  {FILES[k]}: failed from this source")
                    break
        elif kind == "nist":
            z = dest / "gzip.zip"
            if _fetch(where, z, timeout=120):
                try:
                    with zipfile.ZipFile(z) as zf:
                        for k in missing():
                            with zf.open("gzip/" + FILES[k]) as src, open(dest / FILES[k], "wb") as out:
                                shutil.copyfileobj(src, out)
                except Exception as e:
                    print(f"  could not read gzip.zip: {e}")
                z.unlink(missing_ok=True)
        if not missing():
            if verbose:
                print(f"  done in {time.time() - t:.0f}s")
            return label
    raise RuntimeError("Could not download EMNIST from any source (Hugging Face mirrors or NIST). "
                       "Check the internet connection and re-run; files already downloaded are kept.")


def orient(images, labels):
    """Return (upright images, was_transposed). Uses the mean image of digit '1' (label 1):
    upright, its ink is concentrated in a few *columns*; transposed, in a few *rows*."""
    ones = images[labels == 1][:5000].astype(np.float32)
    if len(ones) == 0:
        return images, False
    m = ones.mean(0)
    col, row = m.sum(0), m.sum(1)
    peaked = lambda p: p.max() / (p.mean() + 1e-6)
    if peaked(row) > peaked(col):
        return np.ascontiguousarray(images.transpose(0, 2, 1)), True
    return images, False


def load_emnist(dest_dir, verbose=True):
    """Download (if needed) and read EMNIST byclass. Returns dict with train_x/train_y/test_x/test_y
    (uint8, images N x 28 x 28 upright, white ink on black) plus 'source' and 'transposed'."""
    src = download(dest_dir, verbose=verbose)
    d = {k: read_idx(Path(dest_dir) / n) for k, n in FILES.items()}
    d["train_x"], tr = orient(d["train_x"], d["train_y"])
    d["test_x"], _ = orient(d["test_x"], d["test_y"])
    d["source"], d["transposed"] = src, tr
    if verbose:
        print(f"EMNIST byclass: {len(d['train_y']):,} train / {len(d['test_y']):,} test images, "
              f"{len(CLASSES)} classes | source: {src} | stored transposed: {tr} (fixed)")
    return d


# ------------------------------------------------------------------ model

class CharCNN(nn.Module):
    """Small VGG-style CNN for 28x28 characters (~0.45 M parameters)."""
    def __init__(self, num_classes=len(CLASSES), dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2),       # 14x14
                                      *block(32, 64), *block(64, 64), nn.MaxPool2d(2),      # 7x7
                                      *block(64, 128), nn.MaxPool2d(2))                     # 3x3
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 9, 256),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):            # x: (B, 1, 28, 28) in [0, 1], white ink on black
        return self.head(self.features(x))


def to_input(u8):
    """uint8 (N,28,28) array or tensor -> float (N,1,28,28) in [0,1]."""
    t = torch.as_tensor(u8)
    return t.float().div_(255).unsqueeze(1)


def augment(x, g=None):
    """Random slant/rotation/scale/shift and stroke thickness, on the GPU (x: B,1,28,28)."""
    B, dev = x.shape[0], x.device
    r = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    ang, shear, sc = r(-0.17, 0.17), r(-0.3, 0.3), r(0.85, 1.12)
    tx, ty = r(-0.1, 0.1), r(-0.1, 0.1)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos / sc, (-sin + shear) / sc, tx], 1),
                         torch.stack([sin / sc, cos / sc, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    k = torch.rand(B, device=dev)
    thick = F.max_pool2d(x, 3, 1, 1)                 # dilate: thicker strokes
    thin = -F.max_pool2d(-x, 3, 1, 1)                # erode: thinner strokes
    x = torch.where((k < 0.25).view(B, 1, 1, 1), thick, x)
    x = torch.where((k > 0.85).view(B, 1, 1, 1), torch.maximum(thin, 0.6 * x), x)
    return x


def class_weights(labels, power=0.3):
    """Mild inverse-frequency weights (freq^-power, mean 1) so rare lower-case letters count more."""
    cnt = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64) + 1
    w = cnt ** -power
    return torch.tensor(w / (w * cnt).sum() * cnt.sum(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, u8, device, batch=4096):
    model.eval()
    out = []
    for i in range(0, len(u8), batch):
        out.append(model(to_input(u8[i:i + batch]).to(device)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, len(CLASSES))


def accuracy_report(logits, labels):
    """Overall, per-group and case-insensitive accuracy on EMNIST."""
    pred = logits.argmax(1).numpy()
    y = np.asarray(labels)
    lower = np.array([c.lower() for c in CLASSES])
    groups = {"digits": y < 10, "upper": (y >= 10) & (y < 36), "lower": y >= 36}
    rep = {"accuracy": float((pred == y).mean()),
           "accuracy_case_insensitive": float((lower[pred] == lower[y]).mean()), "n": int(len(y))}
    for g, m in groups.items():
        rep[f"accuracy_{g}"] = float((pred[m] == y[m]).mean()) if m.any() else None
    return rep


def train(data, out_dir, epochs=6, batch_size=512, lr=2e-3, val_frac=0.05, train_limit=None, device=None,
          seed=0, verbose=True):
    """Train CharCNN on EMNIST byclass with checkpoints in `out_dir` (resumes from last.pt).
    Data are kept on the GPU as uint8, so an epoch takes well under a minute on a T4.
    Returns (model, info) where info has history, best_val_acc, minutes, test metrics."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    n = len(data["train_y"])
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if train_limit:
        tr_idx = tr_idx[:train_limit]
    xs = torch.as_tensor(data["train_x"][tr_idx]).to(device)
    ys = torch.as_tensor(data["train_y"][tr_idx].astype(np.int64)).to(device)
    xv, yv = data["train_x"][val_idx], data["train_y"][val_idx]

    model = CharCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = (len(tr_idx) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lossf = nn.CrossEntropyLoss(weight=class_weights(data["train_y"][tr_idx]).to(device), label_smoothing=0.05)
    hist, start, best = [], 1, 0.0
    cfg = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "val_frac": val_frac,
           "train_limit": train_limit, "seed": seed, "train_images": int(len(tr_idx))}
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("cfg") == cfg:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            hist, best, start = ck["hist"], ck["best"], ck["epoch"] + 1
            if verbose:
                print(f"resuming EMNIST training from epoch {ck['epoch']} (best val acc {best:.4f})")
        elif verbose:
            print("last.pt was trained with different settings; starting a new run (old files are overwritten)")
    g = torch.Generator(device=device); g.manual_seed(seed + start)
    for ep in range(start, epochs + 1):
        model.train(); t = time.time(); tot, cnt = 0.0, 0
        order = torch.randperm(len(ys), device=device, generator=g)
        for i in range(steps_per_epoch):
            idx = order[i * batch_size:(i + 1) * batch_size]
            x = augment(xs[idx].float().div_(255).unsqueeze(1))
            with torch.autocast(device.type, enabled=amp):
                loss = lossf(model(x), ys[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        va = accuracy_report(predict_logits(model, xv, device), yv)
        row = {"epoch": ep, "train_loss": round(tot / cnt, 4), "val_acc": round(va["accuracy"], 4),
               "val_acc_case_insensitive": round(va["accuracy_case_insensitive"], 4), "secs": round(time.time() - t, 1)}
        hist.append(row)
        if va["accuracy"] > best:
            best = va["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": best, "classes": CLASSES}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "hist": hist, "best": best, "cfg": cfg}, last)
        if verbose:
            print(f"epoch {ep}/{epochs} | loss {row['train_loss']:.3f} | val acc {row['val_acc']:.4f} "
                  f"(case-insens. {row['val_acc_case_insensitive']:.4f}) | {row['secs']:.0f}s")
    del xs, ys
    best_ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model"])
    test = accuracy_report(predict_logits(model, data["test_x"], device), data["test_y"])
    info = {"history": hist, "best_val_acc": best, "best_epoch": best_ck["epoch"], "cfg": cfg,
            "minutes": sum(h["secs"] for h in hist) / 60, "test": test,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "source": data.get("source"), "transposed": bool(data.get("transposed"))}
    (out / "emnist_classifier.json").write_text(json.dumps(info, indent=2))
    with open(out / "history.csv", "w") as f:
        f.write(",".join(hist[0]) + "\n")
        for h in hist:
            f.write(",".join(str(v) for v in h.values()) + "\n")
    if verbose:
        print(f"EMNIST test accuracy {test['accuracy']:.4f} (case-insensitive {test['accuracy_case_insensitive']:.4f})")
    return model, info


def load_classifier(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CharCNN().to(device); m.load_state_dict(ck["model"]); m.eval()
    return m

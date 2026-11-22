"""Training and evaluation for the CRNN.

The pieces, in the order a training run uses them:
  make_loader       batches of similar width (BucketBatchSampler) + collate_fn
  ctc_loss_on_batch forward pass + CTC loss for one batch
  train_one_epoch   AdamW step per batch, gradient clipping, mixed precision on GPU
  evaluate          greedy decoding -> CER / WER (+ every prediction)
  fit               the epoch loop: LR schedule, best checkpoint, early stopping, resume
"""
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import ModelConfig, TrainConfig
from src.data.charset import Charset
from src.data.iam import collate_fn
from src.models.crnn import CRNN, count_parameters
from src.training.sampler import BucketBatchSampler
from src.utils.metrics import cer, wer


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data

def subset(ds, frac: float, seed: int = 0):
    """Random fraction of a dataset (for quick sweeps). Keeps .widths() working."""
    if frac >= 1.0:
        return ds
    idx = sorted(random.Random(seed).sample(range(len(ds)), max(1, int(len(ds) * frac))))
    sub = Subset(ds, idx)
    sub.widths = lambda: _pick(ds.widths(), idx)
    return sub


def _pick(values, idx):
    return [values[i] for i in idx]


def make_loader(ds, batch_size, shuffle, num_workers=0, seed=0):
    widths = ds.widths()
    sampler = BucketBatchSampler(widths, batch_size, shuffle=shuffle, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return loader, sampler


# ------------------------------------------------------------------ one batch

ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
# zero_infinity: if a label can't fit in the time steps (word too long for its
# width), that sample gives loss 0 instead of inf, so one bad sample can't wreck training.


def ctc_loss_on_batch(model, batch, device, amp=False):
    images = batch["images"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        log_probs = model(images)                         # (T, B, C)
    input_lengths = CRNN.output_lengths(batch["widths"])  # real time steps per image
    # CTC is computed in float32: it's numerically touchy in half precision.
    loss = ctc_loss_fn(log_probs.float(), batch["targets"].to(device), input_lengths, batch["target_lengths"])
    return loss, log_probs, input_lengths


# ------------------------------------------------------------------ epochs

def train_one_epoch(model, loader, optimizer, scaler, device, cfg: TrainConfig, log_every=100):
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    for i, batch in enumerate(loader):
        if cfg.max_batches and i >= cfg.max_batches:
            break
        if batch is None:                               # every image in it was unreadable
            continue
        loss, _, _ = ctc_loss_on_batch(model, batch, device, amp=scaler.is_enabled())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                      # so clipping sees real gradient sizes
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(batch["texts"]); n += len(batch["texts"])
        if log_every and (i + 1) % log_every == 0:
            print(f"    batch {i+1}/{len(loader)}  loss {total/n:.3f}  ({time.time()-t0:.0f}s)")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, charset: Charset, device, amp=False, max_batches=None):
    """Greedy-decode a whole split. Returns dict with loss, cer, wer, predictions."""
    model.eval()
    refs, hyps, total, n = [], [], 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if batch is None:                                # every image in it was unreadable
            continue
        loss, log_probs, lengths = ctc_loss_on_batch(model, batch, device, amp=amp)
        best = log_probs.argmax(dim=2).cpu()             # (T, B)
        for b, L in enumerate(lengths.tolist()):
            hyps.append(charset.ctc_greedy_decode(best[:L, b].tolist()))
        refs += batch["texts"]
        total += loss.item() * len(batch["texts"]); n += len(batch["texts"])
    secs = time.time() - t0
    return {"loss": total / max(n, 1), "cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "n": len(refs), "ms_per_word": 1000 * secs / max(len(refs), 1),
            "refs": refs, "hyps": hyps}


# ------------------------------------------------------------------ checkpoints

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_cer, history,
                    model_cfg: ModelConfig, charset: Charset, bad_epochs=0):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_cer": best_cer, "history": history, "bad_epochs": bad_epochs,
                "model_cfg": asdict(model_cfg), "charset": charset.chars}, tmp)
    tmp.replace(path)   # write then rename: a Colab disconnect can't leave a half-written file


def load_model(path, device=None):
    """Rebuild the CRNN from a checkpoint. Returns (model, charset, checkpoint dict)."""
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck["model_cfg"]; mc["cnn_channels"] = tuple(mc["cnn_channels"])
    charset = Charset(ck["charset"])
    model = CRNN(charset.num_classes, ModelConfig(**mc)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, charset, ck


# ------------------------------------------------------------------ the full run

def fit(datasets, charset: Charset, model_cfg: ModelConfig, cfg: TrainConfig, resume=True, verbose=True):
    """Train on datasets['train'], pick the best epoch by val CER.

    Saves to cfg.out_dir:  best.pt (lowest val CER), last.pt (for resuming),
    history.csv (one row per epoch) and config.json.
    """
    seed_everything(cfg.seed)
    device = get_device()
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    amp = cfg.amp and device.type == "cuda"

    train_ds = subset(datasets["train"], cfg.train_frac, cfg.seed)
    train_loader, train_sampler = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader, _ = make_loader(datasets["val"], cfg.batch_size * 2, False, cfg.num_workers)

    model = CRNN(charset.num_classes, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Halve the LR when val CER stops improving for lr_patience epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=cfg.lr_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best_cer, history, bad = 1, math.inf, [], 0

    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best_cer, history = ck["epoch"] + 1, ck["best_cer"], ck["history"]
        bad = ck.get("bad_epochs", 0)
        print(f"Resuming from epoch {ck['epoch']} (best val CER so far {best_cer:.4f})")

    (out / "config.json").write_text(json.dumps(
        {"model": asdict(model_cfg), "train": {k: str(v) for k, v in asdict(cfg).items()},
         "train_words": len(train_ds), "val_words": len(datasets["val"]),
         "parameters": count_parameters(model), "device": str(device)}, indent=2))
    if verbose:
        print(f"device {device} | {count_parameters(model):,} parameters | "
              f"{len(train_ds)} train / {len(datasets['val'])} val words | {len(train_loader)} batches/epoch")

    for epoch in range(start, cfg.epochs + 1):
        if bad >= cfg.patience:
            print(f"Early stop: val CER hasn't improved for {cfg.patience} epochs"); break
        t0 = time.time()
        train_sampler.set_epoch(epoch)
        if hasattr(train_ds, "set_epoch"):          # Model B: new random augmentations each epoch
            train_ds.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg,
                                     log_every=100 if verbose else 0)
        val = evaluate(model, val_loader, charset, device, amp=amp, max_batches=cfg.max_batches)
        scheduler.step(val["cer"])
        improved = val["cer"] < best_cer
        if improved:
            best_cer, bad = val["cer"], 0
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val["loss"], 4),
               "val_cer": round(val["cer"], 4), "val_wer": round(val["wer"], 4),
               "lr": optimizer.param_groups[0]["lr"], "secs": round(time.time() - t0, 1)}
        history.append(row)
        if improved:
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, scaler, epoch, best_cer,
                            history, model_cfg, charset, bad)
        save_checkpoint(last, model, optimizer, scheduler, scaler, epoch, best_cer, history,
                        model_cfg, charset, bad)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
        if verbose:
            k = max(1, len(val["refs"]) // 4)          # 3 examples spread across the val set
            ex = ", ".join(f"{r!r}->{h!r}" for r, h in list(zip(val["refs"], val["hyps"]))[k::k][:3])
            print(f"epoch {epoch:3d} | train loss {train_loss:.3f} | val loss {val['loss']:.3f} | "
                  f"val CER {val['cer']:.4f} WER {val['wer']:.4f} | lr {row['lr']:.1e} | "
                  f"{row['secs']:.0f}s {'*best*' if improved else ''}\n          e.g. {ex}")
    return {"best_val_cer": best_cer, "history": history, "out_dir": str(out)}


def overfit_one_batch(dataset, charset: Charset, model_cfg=None, steps=200, batch_size=16, lr=1e-3):
    """Sanity check: a correct model + loss should memorise one small batch.
    Returns the list of losses (should fall towards 0) and the final predictions."""
    seed_everything(0)
    device = get_device()
    batch = collate_fn([dataset[i] for i in range(min(batch_size, len(dataset)))])
    model = CRNN(charset.num_classes, model_cfg or ModelConfig(dropout=0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        loss, _, _ = ctc_loss_on_batch(model, batch, device)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        _, log_probs, lengths = ctc_loss_on_batch(model, batch, device)
    best = log_probs.argmax(2).cpu()
    preds = [charset.ctc_greedy_decode(best[:L, b].tolist()) for b, L in enumerate(lengths.tolist())]
    return losses, list(zip(batch["texts"], preds))

"""Shared helpers for the webcam / page-mode tests: rendered handwriting-like words and a tiny CRNN."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_fake_iam import find_fonts, load_font, render_word  # noqa: E402

WORDS = ("the", "could", "hardly", "happen", "without", "orientation", "with", "sea", "bottom", "It",
         "means", "some", "having", "small", "matter", "of", "a", "find", "safer", "stacking")


def word_images(texts=WORDS, size=44, seed=0):
    fonts = find_fonts(limit=3) or [None]
    rng = random.Random(seed)
    return [np.asarray(render_word(t, load_font(fonts[i % len(fonts)], size), rng)) for i, t in enumerate(texts)]


def tiny_checkpoint(path, chars="abcdefghijklmnopqrstuvwxyz"):
    """A training-style checkpoint (random weights) for the export / demo tests."""
    from dataclasses import asdict
    from src.config import ModelConfig
    from src.data.charset import Charset
    from src.models.crnn import CRNN
    torch.manual_seed(0)
    cfg = ModelConfig(cnn_channels=(8, 8, 16, 16, 16), lstm_hidden=16)
    cs = Charset(chars)
    m = CRNN(cs.num_classes, cfg).eval()
    torch.save({"model": m.state_dict(), "optimizer": {}, "epoch": 3, "best_cer": 0.5,
                "model_cfg": asdict(cfg), "charset": cs.chars}, path)
    return m, cs

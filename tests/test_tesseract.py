"""Tesseract baseline: preprocessing, normalization and (if installed) a real recognition."""
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.baselines import tesseract as tess
from src.data.iam import WordItem


def test_normalize_and_preprocess():
    assert tess.normalize("  hel lo\n\x0c") == "hello"
    assert tess.normalize("\u201cHi\u2019") == "\"Hi'"
    img = Image.new("L", (50, 20), 0)
    out = tess.preprocess(img, border=16)
    assert out.size == (82, 52) and out.getpixel((0, 0)) == 255 and out.getpixel((40, 26)) == 0
    assert tess.config() == "--oem 1 --psm 8"


def _word_png(path, text):
    import matplotlib
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 40)
    img = Image.new("L", (40 * len(text), 60), 255)
    ImageDraw.Draw(img).text((4, 4), text, fill=0, font=font)
    img.save(path)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_reads_printed_word_and_caches(tmp_path):
    pytest.importorskip("pytesseract")
    items = []
    for i, w in enumerate(["hello", "seven"]):
        p = tmp_path / f"w{i}.png"; _word_png(p, w)
        items.append(WordItem(f"x-{i}", "x", None, p, w))
    cache = tmp_path / "preds.csv"
    rows, secs = tess.run_cached(items, str(cache), workers=2, chunk=1)
    assert [r["hyp"] for r in rows] == ["hello", "seven"] and secs > 0
    rows2, _ = tess.run_cached(items, str(cache), workers=2)         # second call: all from cache
    assert rows2 == rows and len(cache.read_text().splitlines()) == 3


def test_thread_limit_does_not_leak(monkeypatch):
    """OMP_THREAD_LIMIT=1 is for the tesseract processes only (it can hang PyTorch subprocesses)."""
    import os
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    monkeypatch.setattr(tess, "recognize", lambda path, border=16: "x")
    tess.run([WordItem("a", "a", None, "p", "x")], workers=1, progress=False)
    assert "OMP_THREAD_LIMIT" not in os.environ

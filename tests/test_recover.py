"""Label recovery for a cut-off words.txt (scripts/recover_labels.py)."""
import io
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from src.data.recover import entry_from_png_path, words_txt_line

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_cutoff_copy(tmp_path, keep_frac=0.4, strip_xml=False):
    """Fake IAM with embedded XML; words.txt cut off mid-line like the Kaggle copy."""
    fake = tmp_path / "fake"
    subprocess.run([PY, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake), "--embed-xml",
                    "--writers", "6", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    data = tmp_path / "iam"
    (data / "ascii").mkdir(parents=True)
    shutil.copytree(fake / "words", data / "words")
    if strip_xml:   # same pixels, metadata gone
        for p in (data / "words").rglob("*.png"):
            if p.stat().st_size:
                Image.open(p).convert("L").save(p)
    lines = [l for l in (fake / "ascii" / "words.txt").read_text().splitlines() if not l.startswith("#")]
    k = int(len(lines) * keep_frac)
    (data / "ascii" / "words.txt").write_text("\n".join(lines[:k]) + "\n" + lines[k][:10] + "\n")
    return fake, data, lines


def make_parquet(fake, out, frac=1.0):
    """Stand-in for the Hugging Face parquet: original PNG bytes (with XML) + text."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pngs = sorted(p for p in (fake / "words").rglob("*.png") if p.stat().st_size)
    pngs = pngs[: int(len(pngs) * frac)]
    imgs = [{"bytes": p.read_bytes(), "path": None} for p in pngs]
    texts = [entry_from_png_path(p)["text"] for p in pngs]
    pq.write_table(pa.table({"image": imgs, "text": texts}), out)


def test_xml_rebuilds_original_line(tmp_path):
    fake, _, lines = make_cutoff_copy(tmp_path)
    by_id = {l.split(" ")[0]: l for l in lines}
    for p in list((fake / "words").rglob("*.png"))[:20]:
        e = entry_from_png_path(p)
        if e:
            assert words_txt_line(e) == by_id[e["id"]]


def test_recover_from_own_xml(tmp_path):
    fake, data, lines = make_cutoff_copy(tmp_path)
    out = tmp_path / "words_recovered.txt"
    r = subprocess.run([PY, str(ROOT / "scripts" / "recover_labels.py"), "--data-root", str(data),
                        "--out", str(out), "--hf-parquet", "/nonexistent"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    got = sorted(l for l in out.read_text().splitlines() if not l.startswith("#"))
    empty_img = 1   # make_fake_iam plants one empty (unreadable) image: it can't be recovered
    assert len(got) == len(lines) - empty_img or len(got) == len(lines)
    assert set(got) <= set(lines)          # every rebuilt line equals the true line


def test_recover_from_hf_parquet_when_xml_stripped(tmp_path):
    fake, data, lines = make_cutoff_copy(tmp_path, strip_xml=True)
    make_parquet(fake, tmp_path / "hf.parquet")
    out = tmp_path / "words_recovered.txt"
    r = subprocess.run([PY, str(ROOT / "scripts" / "recover_labels.py"), "--data-root", str(data),
                        "--out", str(out), "--hf-parquet", str(tmp_path / "hf.parquet")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "100.00% identical images" in r.stdout
    assert set(l for l in out.read_text().splitlines() if not l.startswith("#")) <= set(lines)


def test_low_match_rate_writes_nothing(tmp_path):
    fake, data, _ = make_cutoff_copy(tmp_path, keep_frac=0.2, strip_xml=True)
    make_parquet(fake, tmp_path / "hf.parquet", frac=0.3)    # most unlabelled words absent
    out = tmp_path / "words_recovered.txt"
    r = subprocess.run([PY, str(ROOT / "scripts" / "recover_labels.py"), "--data-root", str(data),
                        "--out", str(out), "--hf-parquet", str(tmp_path / "hf.parquet")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and not out.exists()
    assert "labelled subset" in r.stdout

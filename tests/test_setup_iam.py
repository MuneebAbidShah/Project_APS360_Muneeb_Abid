"""setup_iam.py must find IAM in the different layouts people download."""
import importlib.util
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from src.config import DataConfig
from src.data.iam import make_datasets

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_iam", ROOT / "scripts" / "setup_iam.py")
setup_iam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_iam)


def make_fake(tmp_path):
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "3"],
                   check=True, capture_output=True)
    return fake


def check_loads(out):
    ds, stats = make_datasets(DataConfig(iam_root=out, split_dir=None))
    assert stats["kept"] > 0 and stats["bad_or_missing_image"] == 1
    x, *_ = ds["train"][0]
    assert x.shape[1] == 32


def test_kaggle_style_layout(tmp_path):
    # archive.zip -> iam_words/words.txt + iam_words/words/a01/a01-000u/*.png
    fake = make_fake(tmp_path)
    raw = tmp_path / "raw" / "iam_words"
    raw.mkdir(parents=True)
    shutil.copy(fake / "ascii" / "words.txt", raw / "words.txt")
    shutil.copytree(fake / "words", raw / "words")
    setup_iam.setup_from_raw(tmp_path / "raw", tmp_path / "iam")
    check_loads(tmp_path / "iam")


def test_nested_tgz_layout(tmp_path):
    # archive.zip -> words.tgz + ascii.tgz (official archives zipped together)
    fake = make_fake(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    with tarfile.open(raw / "words.tgz", "w:gz") as t:
        for d in (fake / "words").iterdir():
            t.add(d, arcname=d.name)
    with tarfile.open(raw / "ascii.tgz", "w:gz") as t:
        t.add(fake / "ascii" / "words.txt", arcname="words.txt")
        t.add(fake / "ascii" / "forms.txt", arcname="forms.txt")
    setup_iam.setup_from_raw(raw, tmp_path / "iam")
    assert (tmp_path / "iam" / "ascii" / "forms.txt").exists()
    check_loads(tmp_path / "iam")


def test_flat_layout(tmp_path):
    # all pngs in one folder
    fake = make_fake(tmp_path)
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    shutil.copy(fake / "ascii" / "words.txt", raw / "labels_words.txt")
    for p in (fake / "words").rglob("*.png"):
        shutil.copy(p, raw / "images" / p.name)
    setup_iam.setup_from_raw(raw, tmp_path / "iam")
    check_loads(tmp_path / "iam")


def test_kaggle_layout_with_real_ids_uses_aachen_and_writer_table(tmp_path):
    """Like the user's Kaggle download: real form IDs, no forms.txt."""
    fake = tmp_path / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_iam.py"), "--out", str(fake),
                    "--real-form-ids", "--writers", "10", "--forms-per-writer", "1", "--lines-per-form", "2"],
                   check=True, capture_output=True)
    raw = tmp_path / "raw" / "iam_words"
    raw.mkdir(parents=True)
    # cut words.txt off mid-line, like the real Kaggle file
    lines = (fake / "ascii" / "words.txt").read_text().splitlines()
    (raw / "words.txt").write_text("\n".join(lines[:-10]) + "\n" + lines[-10][:12] + "\n")
    shutil.copytree(fake / "words", raw / "words")
    setup_iam.setup_from_raw(tmp_path / "raw", tmp_path / "iam")
    assert not (tmp_path / "iam" / "ascii" / "forms.txt").exists()

    ds, stats = make_datasets(DataConfig(iam_root=tmp_path / "iam"))
    assert stats["split_method"].startswith("split files")
    assert stats["writer_source"] == "bundled iam_form_writers.csv"
    assert stats["malformed_line"] == 1
    writers = {s: {i.writer for i in ds[s].items} for s in ("train", "val", "test")}
    assert None not in set().union(*writers.values())
    assert not (writers["train"] & writers["val"] or writers["train"] & writers["test"]
                or writers["val"] & writers["test"])

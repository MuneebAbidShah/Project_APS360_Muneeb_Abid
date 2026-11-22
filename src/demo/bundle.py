"""Build the self-contained live_demo folder (+ live_demo.zip) for the Windows laptop.

Everything the demo imports is copied unchanged from this codebase, so the laptop runs exactly the
model, charset and preprocessing code used in training and evaluation.
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLED = ["src/__init__.py", "src/config.py", "src/data/__init__.py", "src/data/charset.py",
           "src/data/transforms.py", "src/models/__init__.py", "src/models/crnn.py",
           "src/demo/__init__.py", "src/demo/webcam.py", "src/demo/page_reader.py"]
REQUIREMENTS = """# CPU-only is enough. On Windows, the normal 'torch' from pip is the CPU build.
torch>=2.1
opencv-python>=4.8
numpy>=1.24
pillow>=10.0
# optional, Windows only: lets the demo show camera NAMES (e.g. "DroidCam Video") next to their numbers
pygrabber>=0.2; sys_platform == "win32"
"""


def build(out_dir, model_file, info=None, samples=()):
    """Write live_demo/ into out_dir. `samples` = [(png path, label)] for testing without a camera.
    Only our own files are (over)written; anything else in the folder (e.g. webcam_test/) is kept."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for rel in BUNDLED:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, out / rel)
    shutil.copy(ROOT / "src/demo/live_demo.py", out / "live_demo.py")
    shutil.copy(ROOT / "src/demo/README_live_demo.md", out / "README.md")
    (out / "requirements.txt").write_text(REQUIREMENTS)
    if Path(model_file).resolve() != (out / "crnn_model.pt").resolve():
        shutil.copy(model_file, out / "crnn_model.pt")
    (out / "model_info.json").write_text(json.dumps(info or {}, indent=2))
    if samples:
        (out / "samples").mkdir(exist_ok=True)
        rows = ["file,label"]
        for p, label in samples:
            shutil.copy(p, out / "samples" / Path(p).name)
            rows.append(f"{Path(p).name},{label}")
        (out / "samples" / "labels.csv").write_text("\n".join(rows) + "\n")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def make_zip(folder, zip_path, skip=("webcam_test", "__pycache__")):
    """live_demo.zip with a top-level live_demo/ folder (captures in webcam_test/ are left out)."""
    folder = Path(folder)
    tmp = str(zip_path) + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            rel = p.relative_to(folder)
            if p.is_file() and not (set(rel.parts) & set(skip)):
                z.write(p, f"live_demo/{rel.as_posix()}")
    os.replace(tmp, zip_path)
    return str(zip_path)

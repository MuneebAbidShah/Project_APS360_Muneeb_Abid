"""Generate both Colab notebooks from this repo, so they always share one codebase.

    python tools/build_notebooks.py

Each notebook writes the WHOLE package into /content/aps360_htr. The files that
notebook is about are shown as readable %%writefile cells; everything else is
unpacked from one embedded zip (identical bytes to the repo).
"""
import base64
import io
import json
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]
PROJECT_DIR = ("/content/drive/MyDrive/University (Mech Eng)/Year 3/Fall/"
               "APS360 (Introduction to Machine Learning)/Project/Project Progress Report")
SKIP_PARTS = {"__pycache__", ".pytest_cache", "fake_iam", "outputs", "notebooks", "tools"}


def codebase_files():
    files = []
    for p in sorted(R.rglob("*")):
        rel = p.relative_to(R)
        if p.is_file() and not (set(rel.parts) & SKIP_PARTS) and p.suffix not in (".pyc", ".png", ".tmp"):
            if rel.parts[0] in ("src", "scripts", "tests", "data") or rel.name in ("requirements.txt", "README.md"):
                files.append(rel.as_posix())
    return files


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        s = s.replace("\\n", "\n")        # headings written as "title\\nbody" in this file
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})

    def code(self, s):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": s.strip("\n").splitlines(True)})

    def writefile(self, rel):
        self.code(f"%%writefile /content/aps360_htr/{rel}\n" + (R / rel).read_text())

    def save(self, path):
        for i, c in enumerate(self.cells):
            c["id"] = f"c{i:03d}"
        nb = {"cells": self.cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
              "name": "python3"}, "language_info": {"name": "python"},
              "colab": {"provenance": [], "toc_visible": True}}, "nbformat": 4, "nbformat_minor": 5}
        Path(path).write_text(json.dumps(nb, indent=1))
        print(path, len(self.cells), "cells")


# ------------------------------------------------------------------ shared sections

def mount(nb):
    nb.md("## Mount Google Drive\nYour Drive appears at `/content/drive/MyDrive`. Allow access when Colab asks.")
    nb.code("from google.colab import drive\ndrive.mount('/content/drive')")


def paths(nb, extra=""):
    nb.md("""
## Paths
`PROJECT_DIR` is the Drive folder that holds `archive.zip`. If it has been moved, the cell searches your Drive for it.
Heavy files go on Colab's local disk (`/content`), which is far faster than Drive. Results are saved back to Drive.
""")
    nb.code(f'''
import os, subprocess
PROJECT_DIR = "{PROJECT_DIR}"
if not os.path.exists(os.path.join(PROJECT_DIR, "archive.zip")):
    print("archive.zip not found at PROJECT_DIR; searching Drive (can take a minute)...")
    hits = [h for h in subprocess.run(["find", "/content/drive/MyDrive", "-name", "archive.zip", "-size", "+500M"],
                                      capture_output=True, text=True).stdout.split("\\n") if h]
    assert hits, "Could not find archive.zip anywhere in your Drive"
    PROJECT_DIR = os.path.dirname(hits[0])

ARCHIVE    = os.path.join(PROJECT_DIR, "archive.zip")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")       # results saved back to Drive
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")         # figures/tables for the progress report
OFFICIAL_WORDS_TXT = os.path.join(PROJECT_DIR, "words.txt")            # full official file, if you add it
RECOVERED_WORDS_TXT = os.path.join(PROJECT_DIR, "words_recovered.txt") # made by the label-recovery step
CODE_DIR   = "/content/aps360_htr"                      # project code (this session only)
RAW_DIR    = "/content/iam_raw"                         # archive.zip unpacked here
DATA_ROOT  = "/content/iam"                             # tidy IAM layout (symlinks into RAW_DIR)
os.environ["HTR_DATA_ROOT"] = DATA_ROOT                 # src/config.py reads this
os.makedirs(REPORT_DIR, exist_ok=True)
{extra}print("PROJECT_DIR:", PROJECT_DIR)
''')


def write_code(nb, readable, intro, table):
    files = codebase_files()
    hidden = [f for f in files if f not in readable]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in hidden:
            z.write(R / f, f)
    b64 = base64.b64encode(buf.getvalue()).decode()
    nb.md(f"## Write the project code\n{intro}\n\n{table}")
    nb.md(f"""
### The rest of the codebase (embedded zip)
Every step notebook is generated from one repo; this one embeds code **{CODE_VERSION}** (`aps360_htr_code_{CODE_VERSION}.zip` in your project folder).
This cell unpacks the {len(hidden)} files that aren't shown below (see the other notebook or the zip to read them).
The long string is just that zip, base64-encoded.
""")
    nb.code(f'''
import base64, io, os, shutil, sys, zipfile
# Start from a clean copy every time: delete any older code left in /content/aps360_htr,
# and forget already-imported project modules so this session can't keep using stale code.
os.chdir("/content")
shutil.rmtree("/content/aps360_htr", ignore_errors=True)
for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[m]
os.makedirs("/content/aps360_htr", exist_ok=True)
CODE_B64 = "{b64}"
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall("/content/aps360_htr")
with open("/content/aps360_htr/CODE_VERSION", "w") as f:
    f.write("{CODE_VERSION}\\n")
print("unpacked", len(zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).namelist()), "files | code {CODE_VERSION}")
''')
    nb.code("import os\nfor d in sorted({os.path.dirname(f) for f in %r}):\n    os.makedirs(f'/content/aps360_htr/{d}', exist_ok=True)\n%%cd /content/aps360_htr"
            % readable)
    for group, fs in readable_groups(readable):
        nb.md(group.replace("\\n", "\n"))      # titles are written with a literal backslash-n
        for f in fs:
            nb.writefile(f)


CODE_VERSION = "v16"   # bump together with the uploaded aps360_htr_code_vN.zip

_GROUPS = {}


def readable_groups(readable):
    out = []
    for title, fs in _GROUPS["current"]:
        out.append((title, [f for f in fs if f in readable]))
    return out


def install(nb):
    nb.md("""
## Install missing packages
Colab already has PyTorch, OpenCV, NumPy, Pillow, matplotlib, pyarrow and pytest, so this only installs what's missing.
`editdistance` (faster CER/WER) is optional; the code falls back to pure Python without it.
""")
    nb.code('''
import importlib.util, subprocess, sys
need = {"torch": "torch", "torchvision": "torchvision", "cv2": "opencv-python", "numpy": "numpy",
        "PIL": "pillow", "matplotlib": "matplotlib", "tqdm": "tqdm", "pytest": "pytest",
        "pyarrow": "pyarrow", "huggingface_hub": "huggingface_hub",
        "pytesseract": "pytesseract", "editdistance": "editdistance"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
print("missing:", missing or "nothing")
for pkg in missing:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    note = " (optional: CER/WER use the pure-Python fallback)" if pkg == "editdistance" else ""
    print(f"  {pkg}: {'installed' if r.returncode == 0 else 'install failed' + note}")
''')


def unzip_and_setup(nb):
    nb.md("""
## Unpack archive.zip (about 1.2 GB)
Copied to Colab's disk and unzipped there, which takes a few minutes. Re-running in the same session skips it.
""")
    nb.code('''
import os, shutil, time
marker = os.path.join(RAW_DIR, ".unzipped")
if os.path.exists(marker):
    print("Already unzipped this session:", RAW_DIR)
else:
    t = time.time()
    shutil.copy(ARCHIVE, "/content/archive.zip")
    print(f"copied in {time.time()-t:.0f}s"); t = time.time()
    !unzip -q -o /content/archive.zip -d {RAW_DIR}
    os.remove("/content/archive.zip")
    open(marker, "w").close()
    print(f"unzipped in {time.time()-t:.0f}s")
!ls {RAW_DIR}
''')
    nb.md("""
## Find words.txt and the images
`setup_iam.py` finds the word images and a `words.txt` in any download layout (official tgz, Kaggle `iam_words/`, or a flat folder) and links them into `/content/iam`.

**Which labels get used** (first one that exists wins):
1. `words.txt` in your project folder: the full official file, if you download it from IAM;
2. `words_recovered.txt` in your project folder, made by the label-recovery step below;
3. the `words.txt` inside `archive.zip`. The Kaggle copy is **cut off at e07-072**, so only ~39% of IAM has labels.
""")
    nb.code('''
def pick_words_txt():
    for p in (OFFICIAL_WORDS_TXT, RECOVERED_WORDS_TXT):
        if os.path.exists(p):
            return p
    return None

def run_setup():
    wt = pick_words_txt()
    print("labels from:", wt or "the words.txt inside archive.zip")
    extra = f'--words-txt "{wt}"' if wt else ""
    !python scripts/setup_iam.py --raw {RAW_DIR} --out {DATA_ROOT} {extra}

run_setup()
''')


def recover(nb):
    nb.md("""
## Recover the missing labels (only needed once)
**Why:** the Kaggle `words.txt` stops part-way through line `e07-072-09-`, but `archive.zip` has the images for all of IAM. Without the missing lines, about 60% of the images can't be used.

**How:** every original IAM PNG has its form's XML stored *inside the file* (a PNG text chunk). That XML gives the word's ID, its text, its tag, its bounding box and its line's segmentation status (ok/err), which are exactly the fields of a `words.txt` line. `recover_labels.py`:
1. first tries the XML inside **your own** images (fast, no download);
2. if they've been stripped, it downloads the Hugging Face copy of IAM ([priyank-m/IAM_words_text_recognition](https://huggingface.co/datasets/priyank-m/IAM_words_text_recognition), ~1.2 GB), whose PNGs still carry the XML, and matches **by word ID**. Images whose ID can't be found fall back to a pixel-hash match.

**Checks it prints, for the report:**
- *rebuilt vs original*: for words that have both, the rebuilt line must equal the original Kaggle line (expect ~100%);
- *pixel check*: the HF image is pixel-identical to yours for a sample of recovered words;
- matched / unmatched / ambiguous counts. Below 90%, nothing is written and training uses the labelled subset.

The result is saved to your project folder as `words_recovered.txt` (original lines first, then a `#--- recovered entries below ---` section), so later sessions and the Step 2 notebook reuse it. Recovered words keep their real IAM IDs, so the **Aachen writer-disjoint splits still apply**.
""")
    nb.code('''
if os.path.exists(OFFICIAL_WORDS_TXT):
    print("Using the official full words.txt; no recovery needed.")
elif os.path.exists(RECOVERED_WORDS_TXT):
    print("Reusing", RECOVERED_WORDS_TXT)
else:
    !python scripts/recover_labels.py --data-root {DATA_ROOT} --out "{RECOVERED_WORDS_TXT}" \\
        --hf-cache /content/hf_iam --report "{OUTPUT_DIR}/label_recovery.json" \\
        --preview "{OUTPUT_DIR}/label_recovery_preview.png"
    if os.path.exists(RECOVERED_WORDS_TXT):
        run_setup()          # re-link with the recovered labels
        from IPython.display import Image as _I, display
        display(_I(os.path.join(OUTPUT_DIR, "label_recovery_preview.png")))
''')


def tests(nb, args, blurb):
    nb.md(f"## Run the tests\n{blurb} Every line should say `passed`.")
    nb.code(f"!python -m pytest -q {args}")


# ------------------------------------------------------------------ step 1

def build_step1():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 1 (IAM data pipeline)

This notebook prepares the IAM word images for the CRNN + CTC model. It:
1. mounts your Google Drive and writes the project code into the session;
2. unpacks `archive.zip` (IAM words) and finds `words.txt` and the images, whatever layout the download uses;
3. **recovers the labels missing from the cut-off Kaggle `words.txt`**, once, saving `words_recovered.txt` to your project folder;
4. runs the tests, then filters and splits IAM, prints the numbers for the report, and shows a sample batch.

**Run:** Runtime → Run all. CPU is fine for this step. When Colab asks for Drive access, allow it.
Your Drive files are never changed; the only things added are `outputs/` and `words_recovered.txt`.
""")
    mount(nb); paths(nb)
    _GROUPS["current"] = [
        ("### Package and settings", ["src/__init__.py", "src/data/__init__.py", "src/utils/__init__.py", "src/config.py"]),
        ("### Character set and CTC decoding\nBlank = 0, so real characters start at 1. Greedy decoding takes the best class per time step, merges repeats, then drops blanks.",
         ["src/data/charset.py"]),
        ("### Image preprocessing", ["src/data/transforms.py"]),
        ("### IAM loading, filtering, splitting and batching", ["src/data/iam.py"]),
        ("### Metrics (CER / WER)", ["src/utils/metrics.py"]),
        ("### Label recovery from IAM's embedded XML", ["src/data/recover.py", "scripts/recover_labels.py"]),
        ("### Scripts", ["scripts/setup_iam.py", "scripts/inspect_data.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The data code is shown below as `%%writefile` cells you can read. Model and training code (Step 2) comes from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/config.py` | all settings: data path, image height 32, width limits, filters, model and training defaults |
| `src/data/charset.py` | characters ↔ numbers; CTC blank = 0; greedy CTC decoding |
| `src/data/transforms.py` | resize to 32 px high (keeping aspect ratio), normalize to [-1, 1] |
| `src/data/iam.py` | read `words.txt`, filter bad samples, writer-disjoint splits, `Dataset`, `collate_fn` |
| `src/utils/metrics.py` | CER and WER |
| `src/data/recover.py`, `scripts/recover_labels.py` | rebuild missing `words.txt` lines from IAM's embedded XML |
| `scripts/setup_iam.py` | find `words.txt` and the images in any download layout |
| `scripts/inspect_data.py` | statistics + a picture of one batch |
| `data/` | Aachen split lists + the form → writer table (`iam_form_writers.csv`, see its `_SOURCE.md`) |""")
    install(nb); unzip_and_setup(nb); recover(nb)
    tests(nb, "tests -k 'not training'", "These cover the charset, CTC decoding, CER/WER, padding, the writer table, label recovery and every download layout. The training tests run in Step 2.")
    nb.md("## See the pieces in action")
    nb.code('''
from src.data.charset import Charset
from src.utils.metrics import cer, wer
cs = Charset()
print("classes (78 chars + blank):", cs.num_classes)
print("encode('Hello'):", cs.encode("Hello"))
h, e, l, o = (cs.encode(c)[0] for c in "helo")
frames = [0, h, h, e, 0, l, l, 0, l, o, o, 0]          # what a CTC model outputs per column
print("greedy CTC on", frames, "->", repr(cs.ctc_greedy_decode(frames)))
print("CER(['hello'], ['helo']) =", cer(["hello"], ["helo"]), "(1 deletion / 5 chars)")
print("WER(['a','cat'], ['a','cot']) =", wer(["a", "cat"], ["a", "cot"]))
''')
    nb.md("""
## Inspect the real IAM data
Loads the labels, applies the filters, splits by writer, and draws one batch. Opening all images to catch broken files takes a few minutes.

**For the report:**
- **Filtering:** how many words each filter removed (`segmentation_err` = IAM marked the crop as badly segmented, `chars_outside_charset`, `too_long` = over 32 characters, `bad_or_missing_image`, `malformed_line` = a cut-off `words.txt` line). `not_in_any_split` = words from forms that aren't in the Aachen lists.
- **Splits:** words and writers per split. `writers shared between splits` must be 0. With all of IAM, the Aachen splits have 283 / 56 / 161 writers.
- Characters that never appear in train (e.g. `&`, `+`) are just rare in IAM.
""")
    nb.code('!python scripts/inspect_data.py --data-root {DATA_ROOT} --out "{OUTPUT_DIR}/iam_sample_batch.png" --stats-out "{OUTPUT_DIR}/iam_stats.json"')
    nb.code('from IPython.display import Image, display\ndisplay(Image(os.path.join(OUTPUT_DIR, "iam_sample_batch.png")))')
    nb.md("""
Each row is one word from a padded batch: all 32 px high, and the red line marks where the real image ends and the padding starts.

**Next:** open **APS360 - HTR Step 2 (Train CRNN)** to train the model. It reuses `words_recovered.txt` automatically.
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step1_Data.ipynb")


# ------------------------------------------------------------------ step 2

def build_step2():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 2 (Train the CRNN)

Trains the CRNN + CTC recognizer from the proposal on IAM words, then evaluates it.

| Section | What happens |
|---|---|
| Setup | GPU check, Drive, code, data (reuses `words_recovered.txt` from Step 1) |
| Model code | the CRNN, training loop, evaluation, sweep (readable cells) |
| Sanity check | the model memorises one batch: proves the model, loss and decoding are wired up correctly |
| Sweep | 12 short runs (lr × hidden grid + dropout / batch / hidden / depth variations), ranked by **validation** CER; resumable |
| Train | uses the sweep winner (or the defaults), full run with checkpoints on Drive; re-run to resume |
| Evaluate | CER / WER on val and **test**, speed per word, example predictions, common mistakes |
| Report outputs | every figure, table and number the progress report needs, saved to `outputs/report/` on Drive |

**Before you run:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all.
**Run Step 1 first** (at least once) so `words_recovered.txt` exists; otherwise this trains on the cut-off subset only.
""")
    nb.md("## Check the GPU\nYou should see a Tesla T4. If it says CPU, change the runtime type: training on CPU would take many hours.")
    nb.code('''
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, then Runtime -> Restart and run all.")
''')
    mount(nb)
    paths(nb)   # RUN_NAME / RUN_DIR are set in the Train cell (they include the chosen settings)
    _GROUPS["current"] = [
        ("### The CRNN model\nCNN (5 conv → BatchNorm → ReLU blocks) → 2-layer bidirectional LSTM → linear layer, with log-softmax over the 78 characters + blank. Width shrinks by 4, so a 128-px-wide word gives 32 time steps.",
         ["src/models/__init__.py", "src/models/crnn.py"]),
        ("### Settings (model + training defaults)", ["src/config.py"]),
        ("### Batching words of similar width", ["src/training/__init__.py", "src/training/sampler.py"]),
        ("### Training loop, evaluation, checkpoints", ["src/training/engine.py", "src/training/data.py"]),
        ("### Hyperparameter sweep and picking the winner\n`run_sweep()` (resumable), `choose_settings()` (winner or defaults) and `check_run_dir()` (never resume a checkpoint trained with other settings).",
         ["src/training/sweep.py"]),
        ("### Plots", ["src/utils/plots.py"]),
        ("### Report outputs (figures, LaTeX tables, numbers)\nShared with Step 3. `style.py` sizes every figure for the ICLR text width; `tables.py` writes CSV + booktabs `.tex`; `crnn_report.py` makes each Step 2 output.",
         ["src/report/__init__.py", "src/report/style.py", "src/report/tables.py", "src/report/errors.py",
          "src/report/figures.py", "src/report/model_summary.py", "src/report/predict.py", "src/report/manifest.py",
          "src/report/crnn_report.py", "src/report/compare.py"]),
        ("### Command-line scripts (same functions, for use outside the notebook)",
         ["scripts/train.py", "scripts/evaluate.py", "scripts/sweep.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The model and training code is shown below as `%%writefile` cells. The data code from Step 1 (identical) comes from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/models/crnn.py` | the CRNN; `output_lengths()` gives CTC the real number of time steps per image |
| `src/training/sampler.py` | puts words of similar width in the same batch (less padding, faster) |
| `src/training/engine.py` | `fit()` (AdamW, gradient clipping, mixed precision, LR halving, early stopping, resume), `evaluate()`, checkpoints, `overfit_one_batch()` |
| `src/training/sweep.py` | the 12-run sweep (resumable), winner selection with fallback to defaults, run-folder check |
| `src/training/data.py` | `load_data()`: Step 1's `make_datasets()` + preloading images into RAM |
| `src/utils/plots.py` | training curves and prediction grids |
| `src/report/*.py` | report figures (PNG + PDF), LaTeX tables, error analysis, model summary, `report_numbers.json` |
| `scripts/train.py`, `evaluate.py`, `sweep.py` | command-line versions of the same steps |""")
    install(nb); unzip_and_setup(nb); recover(nb)
    tests(nb, "tests", "All tests, including the model (output shapes, CTC lengths, learning a toy batch) and a 2-epoch training, resume and evaluate run on fake data.")

    nb.md("""
## Load the data
`load_data()` is Step 1's `make_datasets()` (same filters, Aachen splits, writer table), then it **preloads** the train and val images into RAM, resized to 32 px (a few hundred MB for all of IAM, ~2–4 minutes). Colab has only 2 CPU cores, so decoding PNGs every epoch would leave the GPU waiting.

**Broken image files.** A few IAM word PNGs are empty or corrupt (e.g. `a01-117-05-02`, `r06-022-03-05`). They can't crash this cell any more:
- known broken IDs are always dropped (a hardcoded list in `src/data/iam.py`);
- `check_images=False` still checks that every file exists and isn't empty (a few seconds instead of opening all ~115k images);
- any other file that fails to decode during preloading is dropped and listed by name.

All of them are counted as `bad_or_missing_image` in the data-filtering table.
""")
    nb.code('''
from src.training.data import load_data
datasets, charset, stats = load_data(DATA_ROOT, splits=("train", "val"), preload=True, check_images=False)
print({k: stats[k] for k in ("kept", "segmentation_err", "bad_or_missing_image", "malformed_line", "writer_source", "split_method")})
print("bad images dropped:", stats.get("bad_image_ids", []))
''')
    nb.md("""
## Sanity check: memorise one batch
Before a long run, check the model can drive the loss towards 0 on 16 fixed words (no dropout). If it can't, something is wired wrong (labels, CTC lengths, decoding), and training would just waste GPU time. Takes ~30 s on a T4.
""")
    nb.code('''
import matplotlib.pyplot as plt
from src.config import ModelConfig
from src.training.engine import overfit_one_batch
losses, pairs = overfit_one_batch(datasets["train"], charset, ModelConfig(dropout=0.0), steps=300)
plt.figure(figsize=(5, 3)); plt.plot(losses); plt.xlabel("step"); plt.ylabel("CTC loss"); plt.title("One batch"); plt.show()
print(f"loss {losses[0]:.2f} -> {losses[-1]:.3f}")
print("memorised:", sum(r == h for r, h in pairs), "/", len(pairs), pairs[:6])
''')
    nb.md("""
## Hyperparameter sweep
Short runs that compare settings, ranked by **validation** CER only (the test set is never used to choose). The **winner feeds straight into the Train cell below**.

**Design (12 runs, not a full grid):**
- a small grid: learning rate {5e-4, 1e-3, 3e-3} × LSTM hidden size {128, 256} (6 runs);
- one-at-a-time changes around the default (lr 1e-3, hidden 256, dropout 0.3, batch 64, 2 LSTM layers): dropout 0.1 and 0.5, batch 32 and 128, hidden 384, 3 LSTM layers (6 runs).

Every run trains for `SWEEP_EPOCHS = 8` epochs on the same 30% of the training words and is scored on the full validation set. Short runs rank settings by how fast and how well they learn, which is what a sweep can afford; the final model then trains to convergence.

**Time (estimate, not measured on a T4):** about 1.5–3 min per run (batch 32, hidden 384 and 3 layers are the slowest), so roughly **25–45 min** for all 12. After each run the cell prints the time taken and an estimate for the rest.

**Resumable:** each finished run is saved to `outputs/sweep_results.csv` on Drive immediately, and runs already there are skipped. After a disconnect, Run all again and only the unfinished runs train (the interrupted one resumes from its `last.pt`). If all 12 are in the CSV, nothing re-runs. Set `RUN_SWEEP = False` to skip it: then the Train cell still uses an existing `sweep_results.csv`, or the defaults if there is none.
""")
    nb.code('''
from src.training.sweep import run_sweep, sweep_configs, run_name
RUN_SWEEP = True
SWEEP_EPOCHS = 8
SWEEP_TRAIN_FRAC = 0.3
SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")
configs = sweep_configs()
print(f"{len(configs)} sweep settings:")
for c in configs:
    print(f"  {run_name(c):28s} ({c['group']})")
if RUN_SWEEP:
    ranked = run_sweep(datasets, charset, SWEEP_CSV, os.path.join(PROJECT_DIR, "runs", "sweep"), configs,
                       epochs=SWEEP_EPOCHS, train_frac=SWEEP_TRAIN_FRAC)
    print("\\nranked by validation CER (best first):")
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['name']:28s} val CER {r['best_val_cer']:.4f}  WER {r['best_val_wer']:.4f}  "
              f"best epoch {r['best_epoch']}  {r['minutes']:.1f} min")
else:
    print("RUN_SWEEP is False: not training any sweep runs.")
''')
    nb.md("""
## Train
**Settings:** if `sweep_results.csv` exists (from this session or an earlier one), this cell uses the **best row** (lowest val CER, ties broken by val WER). Otherwise, or if no sweep run learned anything (val CER ≥ 99%), it uses the defaults: lr 1e-3, hidden 256, dropout 0.3, batch 64, 2 LSTM layers. It prints which settings it chose and why.

- **Optimizer:** AdamW (weight decay 1e-4). The LR halves after 3 epochs without a val-CER improvement; training stops after 8.
- **Loss:** CTC (blank = 0) with `zero_infinity=True`, so a label that can't fit its image can't produce an infinite loss.
- **Speed:** mixed precision (float16) on the GPU, batches of similar-width words, images preloaded in RAM.
- **Checkpoints** go to `RUN_DIR` on Drive: `best.pt` (lowest val CER), `last.pt` (every epoch), `history.csv`, `config.json`.
- **`RUN_NAME` includes the settings** (e.g. `crnn_lr0.001_h256_d0.3_b64_L2`), so different settings never share a folder. Before training, `check_run_dir()` also compares `config.json` with the current settings and stops with a clear message instead of silently resuming a checkpoint trained differently.
- **If Colab disconnects:** Run all again. The sweep is skipped (already in the CSV), the same winner is picked, and this cell **resumes** from `last.pt`.

**Time (estimate, not yet measured on a T4):** roughly 0.5–1 min per epoch on all of IAM (~48k training words), so ~20–45 min before early stopping (usually 25–40 epochs). The cut-off subset (~31k words) is about a third faster.
""")
    nb.code('''
import json
from src.training.engine import fit
from src.training.sweep import choose_settings, make_configs, check_run_dir
EPOCHS = 40
SETTINGS, SETTINGS_INFO = choose_settings(SWEEP_CSV)
print(f"Settings from: {SETTINGS_INFO['source'].upper()} - {SETTINGS_INFO['reason']}")
for k, v in SETTINGS.items():
    print(f"  {k:12s} = {v:g}" if isinstance(v, float) else f"  {k:12s} = {v}")

RUN_NAME = "crnn_" + run_name(SETTINGS)          # the settings are part of the folder name
RUN_DIR  = os.path.join(PROJECT_DIR, "runs", RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)
model_cfg, train_cfg = make_configs(SETTINGS, epochs=EPOCHS, patience=8, num_workers=0, out_dir=RUN_DIR)
print("RUN_DIR:", RUN_DIR, "|", check_run_dir(RUN_DIR, model_cfg, train_cfg))
json.dump({"settings": SETTINGS, **{k: v for k, v in SETTINGS_INFO.items()}},
          open(os.path.join(RUN_DIR, "settings_choice.json"), "w"), indent=2, default=str)
result = fit(datasets, charset, model_cfg, train_cfg, resume=True)
print("best val CER:", result["best_val_cer"])
''')
    nb.md("## Training curves\nLoss should fall for both train and val. If val loss rises while train loss keeps falling, the model is overfitting; early stopping keeps the best epoch. (Report-quality versions of these curves are saved in the **Report outputs** section below.)")
    nb.code('''
from IPython.display import Image, display
from src.utils.plots import plot_history
png = plot_history(os.path.join(RUN_DIR, "history.csv"), os.path.join(RUN_DIR, "curves.png"))
display(Image(png))
''')
    nb.md("""
## Evaluate on validation and test
`best.pt` is reloaded and greedy-decoded on **val** and then on **test** (unseen writers, Aachen split). Test is used only here, once training is finished.
- **CER / WER** are corpus-level: total edits divided by total reference characters (or words).
- **ms/word** is batched GPU throughput; the single-word latency below is closer to the live webcam case (brief target: under ~1 s per word).
- Every prediction is also saved **with its word ID** (`crnn_val_predictions.csv`, `crnn_test_predictions.csv` in `outputs/report/`), so Step 3 can score Tesseract on exactly the same words.
""")
    nb.code('''
import csv, json, time, torch
from src.training.engine import evaluate, get_device, load_model, make_loader
from src.training.data import load_data
from src.report.predict import predict_dataset, save_predictions
device = get_device()
DEVICE_NAME = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
model, cs, ck = load_model(os.path.join(RUN_DIR, "best.pt"), device)
print(f"best epoch {ck['epoch']}, val CER {ck['best_cer']:.4f}")
from src.training.data import add_dropped
test_sets, _, _ = load_data(DATA_ROOT, splits=("test",), preload=True, check_images=False)
datasets["test"] = test_sets["test"]      # unreadable test images (if any) are dropped here...
add_dropped(stats, test_sets)             # ...and counted as bad images in the data-filtering table
eval_sets = {"val": datasets["val"], "test": test_sets["test"]}
results, preds = {}, {}
os.makedirs(REPORT_DIR, exist_ok=True)
for split, ds in eval_sets.items():
    loader, _ = make_loader(ds, 128, shuffle=False)
    r = evaluate(model, loader, cs, device, amp=device.type == "cuda")
    results[split] = r
    preds[split] = predict_dataset(model, ds, cs, device, batch_size=128, amp=device.type == "cuda")
    save_predictions(preds[split], os.path.join(REPORT_DIR, f"crnn_{split}_predictions.csv"))
    print(f"{split:4s}: {r['n']} words | CER {r['cer']:.4f} | WER {r['wer']:.4f} | {r['ms_per_word']:.2f} ms/word (batched)")

# single-word latency (one image at a time, like the live demo)
x = test_sets["test"][0][0].unsqueeze(0).to(device)
with torch.no_grad():
    for _ in range(5): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
latency_ms = (time.time() - t0) / 50 * 1000
print(f"single-word latency: {latency_ms:.1f} ms on {DEVICE_NAME}")

metrics = {s: {k: r[k] for k in ("cer", "wer", "n", "ms_per_word")} for s, r in results.items()}
metrics["single_word_latency_ms"] = latency_ms
metrics["device"] = DEVICE_NAME
json.dump(metrics, open(os.path.join(RUN_DIR, "metrics.json"), "w"), indent=2)
t = results["test"]
with open(os.path.join(RUN_DIR, "test_predictions.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["reference", "prediction"]); w.writerows(zip(t["refs"], t["hyps"]))
print("saved metrics.json and test_predictions.csv to", RUN_DIR)
''')
    nb.md("""
# Report outputs
Everything the progress report needs is saved to **`outputs/report/`** on Drive (separate from the checkpoints in `runs/`):
figures as **PNG (300 dpi) + PDF**, tables as **CSV + LaTeX booktabs `.tex`** (complete `table` environments you can `\\input{}`),
`report_numbers.json` (every key number in one place) and `report_summary.md` (every file with a suggested caption).

Everything is rebuilt from what is on Drive (`history.csv`, `best.pt`) plus the evaluation above, so it is the same whether training ran in one go or was resumed. Re-running these cells just overwrites the files.

Figures are drawn for the ICLR template's 5.5-inch text width with 8-9 pt fonts: use `\\includegraphics[width=\\linewidth]{fig_....pdf}` (or `0.48\\linewidth` for the half-width ones, side by side).
""")
    nb.code('''
from IPython.display import Image, Markdown, display
from src.report.manifest import Report
from src.report import crnn_report as cr, compare
from src.report.figures import read_history
rep = Report(REPORT_DIR, step=2)
def show(paths):
    for p in paths or []:
        if p.endswith(".png"):
            display(Image(p, width=700))
from src.report.tables import csv_to_markdown
show_table = lambda paths: display(Markdown(csv_to_markdown([p for p in paths if p.endswith(".csv")][0])))
print("report folder:", REPORT_DIR)
''')
    nb.md("### Training curves (from `history.csv`)\n`fig_training_curves` (both panels), `fig_loss_curve` and `fig_val_cer_wer_curve` (half-width). The dotted line is the best epoch, whose weights are in `best.pt`.")
    nb.code('''
curve_paths, history = cr.save_curves(rep, os.path.join(RUN_DIR, "history.csv"))
show(curve_paths[:1])
''')
    nb.md("### Final metrics, model summary and hyperparameters\n`table_crnn_metrics` (quantitative results), `table_model_summary` (layer-by-layer output shapes and parameters: the rubric's *model complexity*) and `table_hyperparameters`.")
    nb.code('''
nums = cr.save_metrics(rep, preds, results, latency_ms, DEVICE_NAME)
layers, total, macs = cr.save_model_summary(rep, model, cs, ck, train_cfg, model_cfg, history, DEVICE_NAME,
                                            settings_info=SETTINGS_INFO)
print(f"{total:,} parameters ({total * 4 / 1e6:.1f} MB as float32), {macs / 1e6:.0f} M multiply-accumulates per 128-px-wide word")
for name in ("table_crnn_metrics", "table_model_summary", "table_hyperparameters"):
    print(name); show_table([os.path.join(REPORT_DIR, name + ".csv")])
''')
    nb.md("### Data statistics and a cleaned sample\n`table_data_filtering` (what each filter removed), `table_data_splits` (words, writers and forms per split), `fig_label_length_hist` (train label lengths) and `fig_preprocessing_examples` (the rubric's *example of a cleaned training sample*: original crop vs. the 32-px CRNN input).")
    nb.code('''
label_source = os.path.basename(pick_words_txt() or "words.txt inside archive.zip")
cr.save_data_stats(rep, stats, datasets, label_source)
show_table([os.path.join(REPORT_DIR, "table_data_filtering.csv")]); show_table([os.path.join(REPORT_DIR, "table_data_splits.csv")])
show(cr.save_label_hist(rep, datasets))
show(cr.save_preprocessing_examples(rep, datasets["train"]))
''')
    nb.md("### Qualitative results\n`fig_crnn_predictions`: 12 **randomly chosen** test words (not cherry-picked) at their original resolution, with the true text and the prediction (green = right, red = wrong).")
    nb.code('''
show(cr.save_prediction_grid(rep, test_sets["test"], preds["test"], n=12, seed=0))
''')
    nb.md("""
### Error analysis
- `fig_crnn_char_confusions`: the most frequent character **substitutions** (e.g. `a → o`), with the total substitutions / deletions / insertions in the title;
- `table_crnn_char_confusions` (all error types, ∅ = missing/extra character) and `table_crnn_word_errors` (most common wrong words);
- `fig_crnn_cer_by_length` + `table_crnn_cer_by_length`: CER per word length (short words and punctuation are often the hardest).
""")
    nb.code('''
conf, breakdown, err_paths = cr.save_error_analysis(rep, preds["test"])
print("test errors:", breakdown)
show(err_paths[:2]); show_table([os.path.join(REPORT_DIR, "table_crnn_word_errors.csv")])
show(cr.save_cer_by_length(rep, preds["test"])[:2])
''')
    nb.md("### Architecture diagram\n`fig_crnn_architecture`: block diagram with the tensor shape after every stage, drawn from the model settings (so it stays correct if you change channels or hidden size).")
    nb.code('''
show(cr.save_architecture(rep, model_cfg, cs.num_classes))
''')
    nb.md("### Sweep table and CRNN vs baselines comparison\n`table_sweep` (every swept setting, best first, the chosen row in bold with a *) and `fig_sweep` (val CER per run, chosen run in orange), if a sweep exists. `table_comparison` + `fig_cer_by_length_comparison` include every baseline that has results (Step 3 Tesseract, Step 4 EMNIST); whichever notebook runs last makes them.")
    nb.code('''
chosen = SETTINGS_INFO["winner"]["name"] if SETTINGS_INFO["source"] == "sweep" else None
show(cr.save_sweep(rep, SWEEP_CSV, chosen_name=chosen))
if os.path.exists(os.path.join(REPORT_DIR, "table_sweep.csv")):
    show_table([os.path.join(REPORT_DIR, "table_sweep.csv")])
if compare.save_comparison(rep):
    show_table([os.path.join(REPORT_DIR, "table_comparison.csv")])
''')
    nb.md("### Summary and list of files\n`report_summary.md` lists every file with a one-line suggested caption; `report_numbers.json` holds every number above.")
    nb.code('''
from src.report.manifest import future_work
summary = rep.write_summary(future_work(rep))   # the EMNIST item drops out once Step 4 has results
display(Markdown(open(summary).read()))
''')
    nb.code('''
print("Saved report files:")
rep.list_files();
''')
    nb.md("""
## Next
- Run **APS360 - HTR Step 3 (Tesseract Baseline)** (CPU runtime is fine) and **Step 4 (EMNIST baseline)** (GPU). Each adds its baseline files to the same `outputs/report/` folder, and the last one to finish rebuilds the **CRNN vs baselines** table with all three models.
- Then upload `outputs/report/` to Overleaf (see `report_summary.md`).
- **Later (augmentation):** webcam-style distortions for training only, then the with/without ablation (`src/data/transforms.py` TODO).
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step2_Train.ipynb")


# ------------------------------------------------------------------ step 3

def build_step3():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 3 (Tesseract baseline)

The proposal's baseline: **Tesseract OCR**, an off-the-shelf recognizer that was *not* trained on IAM, run on the **same IAM val and test words** as the CRNN (same `words_recovered.txt`, same filters, same writer-independent Aachen splits) and scored with the **same CER/WER code**.

| Section | What happens |
|---|---|
| Setup | Drive, code, install Tesseract, data (reuses `words_recovered.txt` from Step 1) |
| Preprocessing check | three test words before/after the white border, with Tesseract's output |
| Time estimate | times 40 words, then estimates the full run; optional subsampling (`N_TEST`) |
| Run | val + test, saved every 500 words (re-run to resume after a disconnect) |
| Report outputs | baseline metrics table, failure grid, pipeline diagram, error analysis and, if Step 2 has run, the **CRNN vs baselines** table (also includes Step 4's EMNIST baseline if it has run) |

**Runtime:** Tesseract runs on the **CPU only**, so a CPU runtime is fine (Runtime → Change runtime type → CPU); a GPU doesn't make it faster.
**Run Step 1 first** (at least once) so `words_recovered.txt` exists. Everything is saved to `outputs/report/` on Drive, next to Step 2's files.
""")
    mount(nb); paths(nb)
    nb.md("""
## Install Tesseract
`tesseract-ocr` is the OCR program (Colab's apt gives version 4.1.x, which has the LSTM engine); `pytesseract` is the Python wrapper that calls it. Takes ~30 s.
""")
    nb.code('''
!apt-get -qq install -y tesseract-ocr > /dev/null
!pip -q install pytesseract
!tesseract --version 2>&1 | head -1
''')
    _GROUPS["current"] = [
        ("### The Tesseract baseline\\nSettings, the light preprocessing and the output normalization are documented at the top of the file.",
         ["src/baselines/__init__.py", "src/baselines/tesseract.py"]),
        ("### Baseline report outputs and the CRNN vs Tesseract comparison",
         ["src/report/tesseract_report.py", "src/report/compare.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The baseline code is shown below as `%%writefile` cells. Everything else (data pipeline, metrics, report helpers) is identical to Steps 1-2 and comes from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/baselines/tesseract.py` | preprocessing (white border), Tesseract call (`--oem 1 --psm 8`), output normalization, parallel + resumable runs |
| `src/report/tesseract_report.py` | baseline metrics table, failure grid, pipeline diagram |
| `src/report/compare.py` | CRNN vs every baseline with results, on exactly the same words (matched by word ID) |
| `src/utils/metrics.py` (zip) | the same CER / WER code used for the CRNN |""")
    install(nb); unzip_and_setup(nb); recover(nb)
    tests(nb, "tests/test_tesseract.py tests/test_report.py tests/test_metrics.py",
          "The baseline (preprocessing, normalization, a real Tesseract call on a printed word, resuming), the report helpers and CER/WER.")

    nb.md("""
## Load the same splits as the CRNN
Same `make_datasets()` as Steps 1-2: same labels, filters and Aachen writer-independent splits. No preloading: Tesseract reads each original PNG itself.
""")
    nb.code('''
from src.training.data import load_data
datasets, charset, stats = load_data(DATA_ROOT, splits=(), preload=False, check_images=False)
print({s: len(datasets[s]) for s in ("val", "test")})
''')
    nb.md("""
## Preprocessing check
Tesseract gets the **original-resolution** word image (not the 32-px version the CRNN uses), with light preprocessing:
- grayscale, plus a **16-px white border** on every side. Tesseract is trained on text with margins; IAM crops touch the ink, and without a border it often returns nothing;
- no resizing, binarization or deskewing: Tesseract binarizes internally, and heavier tuning would no longer be an off-the-shelf baseline.

Settings: `--oem 1` (LSTM engine only), `--psm 8` (treat the image as one word), `-l eng`.
The output is normalized by removing whitespace/newlines and turning curly quotes into ASCII quotes; nothing else is corrected.
""")
    nb.code('''
import matplotlib.pyplot as plt, numpy as np
from PIL import Image as PILImage
from src.baselines import tesseract as tess
print("Tesseract", tess.version(), "|", tess.config())
items = [it for it in datasets["test"].items[:20] if tess.load_image(it.path) is not None][:3]
fig, axes = plt.subplots(len(items), 2, figsize=(8, 1.4 * len(items)))
for (a, b), it in zip(axes, items):
    img = tess.load_image(it.path).convert("L")
    a.imshow(img, cmap="gray"); a.set_title(f"original ({img.width}x{img.height})", fontsize=8)
    b.imshow(tess.preprocess(img), cmap="gray"); b.set_title(f"with border -> Tesseract: {tess.recognize(it.path)!r}  (true: {it.text!r})", fontsize=8)
    for ax in (a, b): ax.axis("off")
plt.tight_layout(); plt.show()
''')
    nb.md("""
## Settings and time estimate
`N_VAL` / `N_TEST` = number of words to run (`None` = the whole split). For a fair comparison, use the full test split if you have the time; otherwise a random subset (e.g. `N_TEST = 2000`) is fine, because the CRNN is then scored on **exactly the same** words in the comparison table.

The cell times 40 words and prints how long each choice would take. Colab's CPU runtime has 2 cores, and each word takes about 0.05-0.1 s per core, so the full test split (~20,000 words) is roughly 10-25 minutes.
""")
    nb.code('''
import os, random, time
N_VAL  = None    # e.g. 2000 for a quicker run; None = all validation words
N_TEST = None    # e.g. 2000; None = all test words
WORKERS = os.cpu_count() or 2

probe = random.Random(1).sample(datasets["test"].items, min(40, len(datasets["test"])))
_, secs = tess.run(probe, workers=WORKERS, progress=False)
ms = 1000 * secs / len(probe)
print(f"{ms:.0f} ms/word with {WORKERS} parallel workers ({ms * WORKERS:.0f} ms per word per core)")
for split in ("val", "test"):
    n_all = len(datasets[split])
    print(f"  {split}: all {n_all:,} words ~{n_all * ms / 60000:.0f} min | 2,000 words ~{2000 * ms / 60000:.1f} min")

def pick(split, n):
    items = datasets[split].items
    return items if not n or n >= len(items) else random.Random(0).sample(items, n)
''')
    nb.md("""
## Run Tesseract on val and test
Results are appended to `outputs/report/tesseract_<split>_predictions.csv` every 500 words. **If Colab disconnects, re-run the notebook**: words already done are skipped. To start over, delete those two CSV files.
""")
    nb.code('''
preds, secs, full_sizes = {}, {}, {}
for split, n in (("val", N_VAL), ("test", N_TEST)):
    items = pick(split, n)
    full_sizes[split] = len(datasets[split])
    print(f"{split}: {len(items):,} of {full_sizes[split]:,} words")
    preds[split], secs[split] = tess.run_cached(items, os.path.join(REPORT_DIR, f"tesseract_{split}_predictions.csv"),
                                                workers=WORKERS, chunk=500)
    refs, hyps = [r["ref"] for r in preds[split]], [r["hyp"] for r in preds[split]]
    from src.utils.metrics import cer, wer
    print(f"  {split}: CER {cer(refs, hyps):.4f} | WER {wer(refs, hyps):.4f} | empty outputs {sum(not h for h in hyps)}")
''')
    nb.md("# Report outputs\nSaved to `outputs/report/` (same folder as Step 2): figures as PNG (300 dpi) + PDF, tables as CSV + LaTeX booktabs `.tex`.")
    nb.code('''
from IPython.display import Image, Markdown, display
from src.report.manifest import Report
from src.report import tesseract_report as tr, crnn_report as cr, compare
from src.report.tables import csv_to_markdown
rep = Report(REPORT_DIR, step=3)
def show(paths):
    for p in paths or []:
        if p.endswith(".png"):
            display(Image(p, width=700))
show_table = lambda name: display(Markdown(csv_to_markdown(os.path.join(REPORT_DIR, name + ".csv"))))
''')
    nb.md("### Baseline metrics\n`table_tesseract_metrics`: CER, WER, how often Tesseract returned nothing, and time per word per CPU core.")
    nb.code('''
tnums = tr.save_metrics(rep, preds, secs, WORKERS, full_sizes, device="CPU")
show_table("table_tesseract_metrics")
''')
    nb.md("### Qualitative: typical failures\n`fig_tesseract_failures`: 12 randomly chosen **wrong** Tesseract predictions (red), at original resolution.")
    nb.code('''
show(tr.save_failure_grid(rep, datasets["test"], preds["test"], n=12, seed=0))
''')
    nb.md("### Error analysis\nSame analysis as for the CRNN: top character substitutions, most common wrong words, CER by word length.")
    nb.code('''
_, tbd, tpaths = cr.save_error_analysis(rep, preds["test"], system="Tesseract", prefix="tesseract")
print("test errors:", tbd)
show(tpaths[:2]); show_table("table_tesseract_word_errors")
show(cr.save_cer_by_length(rep, preds["test"], system="Tesseract", prefix="tesseract")[:2])
''')
    nb.md("### Baseline pipeline diagram\n`fig_tesseract_pipeline`.")
    nb.code('''
show(tr.save_pipeline(rep))
''')
    nb.md("""
### CRNN vs the baselines
`table_comparison` (+ `fig_cer_by_length_comparison`): the CRNN, Tesseract and (if Step 4 has run) the EMNIST baseline on **the same words** (matched by word ID). Needs Step 2's `crnn_metrics.json` and `crnn_*_predictions.csv` in `outputs/report/`. If Step 2 hasn't run yet, this says so, and the last notebook to finish makes the table.
""")
    nb.code('''
if compare.save_comparison(rep):
    show_table("table_comparison")
    show([os.path.join(REPORT_DIR, "fig_cer_by_length_comparison.png")])
''')
    nb.md("### Summary and list of files")
    nb.code('''
from src.report.manifest import future_work
summary = rep.write_summary(future_work(rep))   # the EMNIST item drops out once Step 4 has results
display(Markdown(open(summary).read()))
''')
    nb.code('''
print("Saved report files:")
rep.list_files();
''')
    nb.md("""
## Notes for the report
- Tesseract is a **pretrained printed-text OCR** with no IAM training, so a high CER on cursive handwriting is expected; that gap is the point of the baseline.
- Report the settings (`--oem 1 --psm 8`, 16-px border, the normalization) so the baseline is reproducible.
- The second baseline (EMNIST character classifier) is **Step 4**; run it to add it to the comparison.
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step3_Tesseract.ipynb")


# ------------------------------------------------------------------ step 4

def build_step4():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 4 (EMNIST character-classifier baseline)

The proposal's second baseline: **cut each IAM word into characters with simple rules, then classify every character with a small CNN trained only on EMNIST** (handwritten single characters). It never sees IAM during training, and it can't use context between letters, which is exactly what the CRNN + CTC model adds. It runs on the **same IAM val and test words** as the CRNN and Tesseract (same labels, filters, writer-independent Aachen splits, same broken-image skipping) and is scored with the **same CER/WER code** (case-sensitive).

| Section | What happens |
|---|---|
| Setup | Drive, code, data (reuses `words_recovered.txt` from Step 1) |
| EMNIST | download (no sign-in; mirrors tried in turn), fix the transposed images, preview |
| Train the character CNN | ~0.45 M parameters, 6 epochs, checkpoints on Drive (re-run to resume) |
| Letter splitter | binarize, slant correction, connected components, projection-profile cuts, EMNIST-style 28x28 crops |
| Tune on validation | character width and the IAM character prior, chosen by **val** CER |
| Run | val + test (optional `N_TEST`), predictions saved to `outputs/report/` |
| Report outputs | classifier accuracy, baseline metrics, segmentation examples, failure grid, pipeline diagram, error analysis, and the **CRNN vs Tesseract vs EMNIST** table |

**Runtime: GPU (T4)** (Runtime → Change runtime type → T4 GPU): training takes a few minutes on a T4 but ~30+ min on a CPU. The splitter itself runs on the CPU.
**Time (estimate, not measured on Colab):** ~4–6 min setup (IAM unzip, same as the other steps), ~1 min EMNIST download, ~3–6 min training, ~1–2 min tuning, ~3–5 min for all val + test words: **about 15–20 min in total**.
**Run Step 1 first** (at least once) so `words_recovered.txt` exists. Steps 2 and 3 don't need to be re-run: this notebook rebuilds the comparison table with every model that has results.
""")
    mount(nb); paths(nb)
    _GROUPS["current"] = [
        ("### EMNIST data and the character CNN\\nDownload with fallbacks, the transposition fix, the model and resumable training.",
         ["src/baselines/__init__.py", "src/baselines/emnist.py"]),
        ("### The rule-based letter splitter\\nEvery rule (and how punctuation EMNIST doesn't have is handled) is documented at the top of the file.",
         ["src/baselines/char_segment.py", "src/baselines/emnist_pipeline.py"]),
        ("### Report outputs and the three-way comparison",
         ["src/report/emnist_report.py", "src/report/compare.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The baseline code is shown below as `%%writefile` cells. Everything else (data pipeline, metrics, report helpers) is identical to Steps 1-3 and comes from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/baselines/emnist.py` | EMNIST download (Hugging Face mirrors, then NIST) and checks, orientation fix, CNN, GPU training with checkpoints |
| `src/baselines/char_segment.py` | letter splitter (binarize, deslant, components, cuts, merges), punctuation rules, 28x28 normalization, decoding |
| `src/baselines/emnist_pipeline.py` | runs the splitter + CNN on IAM words, tuning on val, timing, skipping unreadable images |
| `src/report/emnist_report.py` | classifier table, metrics/tuning tables, segmentation figure, failure grid, pipeline diagram |
| `src/report/compare.py` | CRNN vs every baseline with results, on exactly the same words |""")
    install(nb); unzip_and_setup(nb); recover(nb)
    tests(nb, "tests/test_emnist.py tests/test_report.py tests/test_metrics.py",
          "EMNIST reading, orientation and download fallbacks, the CNN and resumable training, the splitter and normalization, decoding, the IAM run (broken images skipped), the report files and the three-way comparison.")

    nb.md("""
## GPU check
Training the character CNN uses the GPU. On a CPU runtime it still works, just slower (switch with Runtime → Change runtime type).
""")
    nb.code('''
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
print("device:", DEVICE_NAME)
if device.type != "cuda":
    print("No GPU: training will take ~30+ min. Consider Runtime -> Change runtime type -> T4 GPU.")
''')
    nb.md("""
## EMNIST (byclass)
**Why byclass:** it is the only EMNIST split that keeps **upper- and lower-case letters apart** (62 classes: 0-9, A-Z, a-z), which our case-sensitive CER needs. `balanced` (47 classes) merges 15 lower-case letters such as c/o/s/x into their capitals, so it could never output those lower-case letters.

**Download, no sign-in:** the original NIST files are fetched from a Hugging Face mirror (two are tried) and, if both fail, from NIST itself (562 MB). Every file is checked (gzip, header, number of images) before use. The four files (~200 MB) are cached in your project folder (`emnist/`), so later sessions skip the download.

**Transposed images:** EMNIST stores every image transposed (rows and columns swapped). The code checks this on the digit '1' (upright it is a vertical stroke) and fixes it; the preview below should show upright characters.
""")
    nb.code('''
import os, shutil, numpy as np, matplotlib.pyplot as plt
from src.baselines import emnist as E
EMNIST_DRIVE = os.path.join(PROJECT_DIR, "emnist")       # cache on Drive (~200 MB)
EMNIST_LOCAL = "/content/emnist"
os.makedirs(EMNIST_LOCAL, exist_ok=True)
if os.path.isdir(EMNIST_DRIVE):                          # copy the cached files to the fast local disk
    for f in E.FILES.values():
        if os.path.exists(os.path.join(EMNIST_DRIVE, f)) and not os.path.exists(os.path.join(EMNIST_LOCAL, f)):
            shutil.copy(os.path.join(EMNIST_DRIVE, f), EMNIST_LOCAL)
emnist = E.load_emnist(EMNIST_LOCAL)
if emnist["source"] != "cached":                         # first time: keep a copy on Drive
    os.makedirs(EMNIST_DRIVE, exist_ok=True)
    for f in E.FILES.values():
        shutil.copy(os.path.join(EMNIST_LOCAL, f), EMNIST_DRIVE)
    print("cached on Drive:", EMNIST_DRIVE)
fig, axes = plt.subplots(2, 12, figsize=(10, 2))
for ax, i in zip(axes.ravel(), np.random.default_rng(0).choice(len(emnist["train_y"]), 24, replace=False)):
    ax.imshow(emnist["train_x"][i], cmap="gray"); ax.set_title(E.CLASSES[emnist["train_y"][i]], fontsize=9); ax.axis("off")
plt.suptitle("EMNIST samples after the orientation fix (should be upright, white on black)", fontsize=9)
plt.tight_layout(); plt.show()
''')
    nb.md("""
## Train the character CNN
A small VGG-style CNN (5 conv layers, ~0.45 M parameters) on 95% of EMNIST byclass training images (5% held out for validation), for `EPOCHS` epochs. The images stay on the GPU, so an epoch should take well under a minute on a T4. Training uses:
- light **augmentation** (slant, rotation, scale, shift, thinner/thicker strokes), because IAM letters are slanted and written with different pens;
- mildly **re-weighted classes** (byclass has ~10x more digits than some lower-case letters) and label smoothing.

Checkpoints (`last.pt` after every epoch, `best.pt` by validation accuracy) are saved to `runs/emnist_cnn/` on Drive. **If Colab disconnects, re-run: training resumes** from the last finished epoch. Changing `EPOCHS`/`TRAIN_LIMIT` starts a new run.
""")
    nb.code('''
EPOCHS = 6
TRAIN_LIMIT = None        # e.g. 200_000 for a quicker run; None = all ~663k training images
EMNIST_RUN = os.path.join(PROJECT_DIR, "runs", "emnist_cnn")
clf_model, clf_info = E.train(emnist, EMNIST_RUN, epochs=EPOCHS, train_limit=TRAIN_LIMIT, device=device)
t = clf_info["test"]
print(f"EMNIST test accuracy {t['accuracy']:.4f} | case-insensitive {t['accuracy_case_insensitive']:.4f} | "
      f"digits {t['accuracy_digits']:.3f} upper {t['accuracy_upper']:.3f} lower {t['accuracy_lower']:.3f}")
''')
    nb.md("""
## Load the same IAM splits as the CRNN
Same `make_datasets()` as Steps 1-3 (same labels, filters, Aachen writer-independent splits, and the same broken-image list). The splitter reads each original PNG itself; an image that can't be decoded is skipped and listed, exactly like the other models.
""")
    nb.code('''
from src.training.data import load_data
from src.baselines import char_segment as cs, emnist_pipeline as EP
datasets, charset, stats = load_data(DATA_ROOT, splits=(), preload=False, check_images=False)
print({s: len(datasets[s]) for s in ("train", "val", "test")})
# IAM character prior: how often each of the 62 characters occurs in the IAM *training* labels,
# relative to how often the classifier saw it in EMNIST (used only if it helps on validation).
PRIOR = cs.char_prior([it.text for it in datasets["train"].items], emnist["train_y"])
''')
    nb.md("""
## The letter splitter on a few validation words
Red boxes = crops sent to the classifier, blue = small marks labelled by the punctuation rules. Boxes are drawn on the **slant-corrected** word. Below each word: the normalized 28x28 crops.

**Characters EMNIST doesn't have** (punctuation): EMNIST only knows 0-9, A-Z, a-z. Small marks are labelled by size/position rules instead: `.` (dot on the baseline), `,` (taller mark at the baseline), `'` (mark above the letters), `"` (two marks), `-` (flat mark in the middle), and for one-mark word images also `:` `;` `!`. Other symbols (`# & ( ) * + / ?`) are never output, so they always count as errors. That is a real limitation of this baseline, and the report should say so.
""")
    nb.code('''
import random
from src.report.predict import original_image
def preview(items, params, alpha=0.0):
    fig, axes = plt.subplots(len(items), 2, figsize=(10, 1.3 * len(items)), gridspec_kw={"width_ratios": [1, 1.4]})
    for (a, b), it in zip(np.atleast_2d(axes), items):
        g = cs.load_gray(it.path)
        pieces, crops, info = cs.segment_word(g, params, return_info=True)
        lp = EP.classify([{"crops": crops}], clf_model, device)[0][0]
        a.imshow(cs.deslanted_gray(g, info["shear"]), cmap="gray")
        for q in pieces:
            a.add_patch(plt.Rectangle((q["x0"], q["y0"]), q["x1"] - q["x0"], q["y1"] - q["y0"], fill=False,
                                      ec="r" if q["kind"] == "char" else "b"))
        a.set_title(f"true {it.text!r} -> {cs.decode(pieces, lp, PRIOR, alpha)!r}", fontsize=8); a.axis("off")
        b.imshow(np.hstack(list(crops)) if len(crops) else np.zeros((28, 28)), cmap="gray"); b.axis("off")
    plt.tight_layout(); plt.show()
preview(random.Random(3).sample(datasets["val"].items, 4), cs.SegParams())
''')
    nb.md("""
## Tune on validation words (never on test)
Two settings are chosen by **validation CER** on `N_TUNE` random validation words:
- the **width factor**: the expected character width as a multiple of the pen's stroke width (on IAM training words, width / characters / stroke width has a median of about 7-8); a blob wider than ~1.5 expected widths is cut;
- the **IAM character prior** on/off: the classifier's log-probabilities plus log(frequency in IAM *training* labels / frequency in EMNIST). It mostly fixes case and digit/letter confusions (`o` vs `O` vs `0`), because IAM text is mostly lower-case letters.
""")
    nb.code('''
N_TUNE = 2000
tune_items = random.Random(0).sample(datasets["val"].items, min(N_TUNE, len(datasets["val"])))
SEG_PARAMS, ALPHA, TUNING = EP.tune(tune_items, clf_model, device, PRIOR)
print(f"chosen: width factor {SEG_PARAMS.width_factor} x stroke width, IAM prior {'on' if ALPHA else 'off'} "
      f"(val CER {TUNING[0]['cer']:.4f} on {len(tune_items):,} words)")
''')
    nb.md("""
## Settings and time estimate
`N_VAL` / `N_TEST` = number of words to run (`None` = the whole split). Use the full splits if you can: the comparison table always scores every model on the words they all have, so a subset also stays fair. The cell times 300 words first.
""")
    nb.code('''
N_VAL  = None     # e.g. 2000; None = all validation words
N_TEST = None     # e.g. 5000; None = all test words
probe = random.Random(1).sample(datasets["test"].items, min(300, len(datasets["test"])))
_, pinfo = EP.run(probe, clf_model, device, SEG_PARAMS, PRIOR, ALPHA, progress=False)
ms = pinfo["ms_per_word"]
print(f"{ms:.1f} ms/word (segmentation {pinfo['seg_ms_per_word']:.1f} ms on the CPU + classification)")
for split in ("val", "test"):
    print(f"  {split}: all {len(datasets[split]):,} words ~{len(datasets[split]) * ms / 60000:.1f} min")
def pick(split, n):
    items = datasets[split].items
    return items if not n or n >= len(items) else random.Random(0).sample(items, n)
''')
    nb.md("""
## Run on val and test
Predictions are saved to `outputs/report/emnist_<split>_predictions.csv` (word ID, writer, truth, prediction, number of pieces).
""")
    nb.code('''
preds, infos, full_sizes = {}, {}, {}
for split, n in (("val", N_VAL), ("test", N_TEST)):
    items = pick(split, n)
    full_sizes[split] = len(datasets[split])
    print(f"{split}: {len(items):,} of {full_sizes[split]:,} words")
    preds[split], infos[split] = EP.run(items, clf_model, device, SEG_PARAMS, PRIOR, ALPHA)
    EP.save_rows(preds[split], os.path.join(REPORT_DIR, f"emnist_{split}_predictions.csv"))
    s = EP.scores(preds[split])
    print(f"  {split}: CER {s['cer']:.4f} | WER {s['wer']:.4f} | case-insensitive CER {s['cer_case_insensitive']:.4f} "
          f"| {infos[split]['ms_per_word']:.1f} ms/word")
''')
    nb.md("# Report outputs\\nSaved to `outputs/report/` (same folder as Steps 2-3): figures as PNG (300 dpi) + PDF, tables as CSV + LaTeX booktabs `.tex`.")
    nb.code('''
from IPython.display import Image, Markdown, display
from src.report.manifest import Report, future_work
from src.report import emnist_report as er, crnn_report as cr, compare
from src.report.tables import csv_to_markdown
rep = Report(REPORT_DIR, step=4)
def show(paths):
    for p in paths or []:
        if p.endswith(".png"):
            display(Image(p, width=700))
show_table = lambda name: display(Markdown(csv_to_markdown(os.path.join(REPORT_DIR, name + ".csv"))))
''')
    nb.md("### The character classifier on its own\\n`table_emnist_classifier`: EMNIST test accuracy (overall, case-insensitive, digits / upper / lower), size and training time.")
    nb.code('''
er.save_classifier(rep, clf_info)
show_table("table_emnist_classifier")
''')
    nb.md("### Baseline metrics and tuning\\n`table_emnist_metrics`: CER/WER on val and test (case-sensitive, like the other models), case-insensitive CER, time per word. `table_emnist_tuning`: the validation grid (chosen row in bold).")
    nb.code('''
settings = {**cs.params_dict(SEG_PARAMS), "iam_prior": bool(ALPHA), "n_tune": len(tune_items)}
enums = er.save_metrics(rep, preds, infos, full_sizes, TUNING, settings, DEVICE_NAME, clf_info)
show_table("table_emnist_metrics"); show_table("table_emnist_tuning")
print("CER on words with only letters/digits (test):", enums["test"]["cer_letters_digits_only_words"])
''')
    nb.md("### Segmentation examples\\n`fig_emnist_segmentation`: 4 randomly chosen test words (not cherry-picked) with the character boxes, the 28x28 crops and the predicted characters.")
    nb.code('''
ex = er.segmentation_examples(datasets["test"], infos["test"]["segs"], infos["test"]["logps"], PRIOR, ALPHA, n=4, seed=0)
show(er.save_segmentation_figure(rep, ex))
''')
    nb.md("### Qualitative: typical failures\\n`fig_emnist_failures`: 12 randomly chosen **wrong** predictions (red), at original resolution.")
    nb.code('''
show(er.save_failure_grid(rep, datasets["test"], preds["test"], n=12, seed=0))
''')
    nb.md("### Error analysis\\nSame analysis as for the CRNN and Tesseract: top character substitutions, most common wrong words, CER by word length.")
    nb.code('''
_, ebd, epaths = cr.save_error_analysis(rep, preds["test"], system="EMNIST baseline", prefix="emnist")
print("test errors:", ebd)
show(epaths[:2]); show_table("table_emnist_word_errors")
show(cr.save_cer_by_length(rep, preds["test"], system="EMNIST baseline", prefix="emnist")[:2])
''')
    nb.md("### Baseline pipeline diagram\\n`fig_emnist_pipeline`.")
    nb.code('''
show(er.save_pipeline(rep))
''')
    nb.md("""
### CRNN vs Tesseract vs EMNIST baseline
`table_comparison` (+ `fig_cer_by_length_comparison`): every model with results, on **the same words** (matched by word ID). Needs Step 2's results (and Step 3's for Tesseract) in `outputs/report/`; whichever notebook runs last rebuilds it with all of them.
""")
    nb.code('''
if compare.save_comparison(rep):
    show_table("table_comparison")
    show([os.path.join(REPORT_DIR, "fig_cer_by_length_comparison.png")])
''')
    nb.md("### Summary and list of files")
    nb.code('''
summary = rep.write_summary(future_work(rep))
display(Markdown(open(summary).read()))
''')
    nb.code('''
print("Saved report files:")
rep.list_files();
''')
    nb.md("""
## Notes for the report
- The EMNIST classifier is good at **isolated** characters (its own test accuracy is in `table_emnist_classifier`), but on IAM words the errors are dominated by **segmentation**: cursive letters are joined, so cuts fall inside letters or miss boundaries, and every wrong cut costs insertions/deletions. That gap is the motivation for a segmentation-free CRNN + CTC model.
- **Case:** the main CER/WER are case-sensitive, exactly like the CRNN and Tesseract. EMNIST can't tell some upper/lower-case pairs apart (o/O, s/S, c/C...), so the case-insensitive CER is reported as an extra.
- **Punctuation:** handled by position/size rules for `. , ' " - : ; !`; the other symbols can't be produced (EMNIST has no punctuation).
- **Tuning was done on validation words only**; the test set was run once with the chosen settings.
- Neither baseline was trained on IAM; the CRNN was.
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step4_EMNIST.ipynb")

# ------------------------------------------------------------------ step 5

def build_step5():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 5 (webcam live demo + webcam test set)

Colab can't open your laptop's camera, so the **live demo runs on your Windows laptop** (CPU), with your phone as the webcam through DroidCam. This notebook:

| Section | What happens |
|---|---|
| Setup | Drive, code (v16), tests |
| Export the CRNN | your trained `runs/crnn_*/best.pt` → a small CPU file (`crnn_model.pt`: float32 weights + model settings + charset + preprocessing settings), checked to give identical outputs |
| Check: demo = evaluation | ~200 IAM test words run through the demo's own code, compared with Step 2's saved test predictions |
| Build `live_demo` | a self-contained folder on Drive + `live_demo.zip`: model, `live_demo.py`, `requirements.txt`, README with Windows steps, the exact training preprocessing code, 3 sample words |
| Load Model B | `outputs/model_b/crnn_model_b.pt` from Step 6 (augmented CRNN), checked to have **exactly Model A's architecture**, charset and preprocessing |
| Score the webcam test set | reads `webcam_test/` (your captured words + `labels.csv`) and runs **Model A, Model B, Tesseract and the EMNIST baseline on the same words**; 4-row table + per-model predictions + figures in `outputs/report/` |

**Runtime: CPU is fine** (no GPU needed; Runtime → Change runtime type → CPU saves GPU quota). No IAM unzip in this step.
**Time (estimate, not measured on Colab):** ~1–2 min setup + tests, ~1 min export and check, under 1 min to build the zip; scoring ~150 webcam words ~1–2 min (Tesseract install ~30 s). **About 5 min.**

**Order:** run this once now (it builds `live_demo.zip`). Collect the webcam words on your laptop, upload the `webcam_test` folder to the project folder, then **Run all again**: the scoring section runs when it finds `webcam_test/`.
""")
    mount(nb); paths(nb)
    _GROUPS["current"] = [
        ("### Webcam photo → IAM-like word, and the CPU model\\nThe same file runs in the live demo and in the scoring below.",
         ["src/demo/__init__.py", "src/demo/webcam.py"]),
        ("### The live demo (runs on your laptop)\\nWORD mode (guide box, `c` to capture) and PAGE mode (beta).",
         ["src/demo/live_demo.py", "src/demo/page_reader.py"]),
        ("### Building the live_demo folder, scoring, report outputs",
         ["src/demo/bundle.py", "src/demo/score.py", "src/report/webcam_report.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The Step 5 code is shown below as `%%writefile` cells. The model, charset and preprocessing code it reuses are identical to Steps 1-4 and come from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/demo/webcam.py` | photo clean-up (lighting, ink, tight crop, IAM-like contrast), CPU export, the `Recognizer` (training transform + greedy CTC) |
| `src/demo/live_demo.py` | the laptop app: camera detection (DirectShow on Windows), guide box, overlay, smoothing, keys, captures to `webcam_test/` |
| `src/demo/page_reader.py` | PAGE mode (beta): page finding/straightening, line + word splitting, position-based tracking, transcript |
| `src/demo/bundle.py` | builds the `live_demo` folder and zip |
| `src/demo/score.py` | reads `webcam_test/`, runs CRNN Model A / Model B / Tesseract / EMNIST on the same cleaned words; Model A vs B architecture check |
| `src/report/webcam_report.py` | 4-row webcam table, clean-up ablation table, per-model prediction CSVs, prediction grids, preprocessing figure, IAM-vs-webcam figure |""")
    install(nb)
    tests(nb, "tests/test_webcam.py tests/test_webcam_models.py tests/test_camera_switch.py tests/test_mjpeg.py tests/test_page_reader.py tests/test_metrics.py tests/test_report.py",
          "Photo clean-up, export (identical outputs), the demo's keys/captures, the self-contained bundle (run headless), the webcam scoring and report files (Model A + Model B + both baselines, 4-row table, per-model predictions, same-architecture check), and PAGE mode: one commit per word over 30 frames, nothing added while pausing, new words appended, no duplicates under jitter/shifts, growing words updated, the line/word splitter on a synthetic page.")
    nb.md("""
## Find the trained CRNN
This is **Model A** (no augmentation). Uses the newest `runs/crnn_*/best.pt` in your project folder (Step 2 trains `crnn_lr0.001_h256_d0.3_b32_L2`), **ignoring Model B's `runs/crnn_aug_*`** (Step 6). Set `CRNN_RUN` to pick another run.
""")
    nb.code('''
import glob, json, os, random, shutil, subprocess, sys, time
import cv2, numpy as np, torch
CRNN_RUN = None          # e.g. "crnn_lr0.001_h256_d0.3_b32_L2"; None = newest run with a best.pt
if CRNN_RUN:
    BEST_PT = os.path.join(PROJECT_DIR, "runs", CRNN_RUN, "best.pt")
else:
    runs = sorted((p for p in glob.glob(os.path.join(PROJECT_DIR, "runs", "crnn_*", "best.pt"))
                   if not os.path.basename(os.path.dirname(p)).startswith("crnn_aug_")), key=os.path.getmtime)
    assert runs, "No runs/crnn_*/best.pt (Model A) found: run Step 2 first."
    BEST_PT = runs[-1]
RUN_NAME = os.path.basename(os.path.dirname(BEST_PT))
CRNN_METRICS = json.load(open(os.path.join(REPORT_DIR, "crnn_metrics.json"))) if os.path.exists(os.path.join(REPORT_DIR, "crnn_metrics.json")) else {}
test = CRNN_METRICS.get("test", {})
assert not RUN_NAME.startswith("crnn_aug_"), "CRNN_RUN must be Model A (Step 2), not Model B"
print("CRNN Model A run:", RUN_NAME, f"({os.path.getsize(BEST_PT) / 1e6:.0f} MB with optimizer state)")
if test:
    print(f"IAM test CER {100 * test['cer']:.2f}% | WER {100 * test['wer']:.2f}% ({test['words']:,} words)")
''')
    nb.md("""
## Export a CPU-friendly model
`best.pt` also holds the optimizer and scheduler state (~60 MB). The export keeps only what the demo needs: float32 weights (~21 MB), the model settings, the 78-character charset and the preprocessing settings (32 px high, width 16-512). It loads with `torch.load(..., weights_only=True)` and needs no GPU. The check below runs the same random input through the original checkpoint and the export.
""")
    nb.code('''
from src.demo import webcam as W, bundle
from src.training.engine import load_model
LOCAL_LIVE = "/content/live_demo"
shutil.rmtree(LOCAL_LIVE, ignore_errors=True); os.makedirs(LOCAL_LIVE)
MODEL_INFO = {"run": RUN_NAME, "iam_test_cer": test.get("cer"), "iam_test_wer": test.get("wer"),
              "code_version": open("/content/aps360_htr/CODE_VERSION").read().strip(),
              "exported": time.strftime("%Y-%m-%d %H:%M")}
exp = W.export_model(BEST_PT, os.path.join(LOCAL_LIVE, "crnn_model.pt"), MODEL_INFO)
MODEL_INFO.update(exp["info"])
ref_model, charset, _ = load_model(BEST_PT, torch.device("cpu"))
rec = W.Recognizer(os.path.join(LOCAL_LIVE, "crnn_model.pt"))
x = torch.rand(2, 1, 32, 160) * 2 - 1
with torch.no_grad():
    diff = (ref_model(x) - rec.model(x)).abs().max().item()
print(f"exported crnn_model.pt: {os.path.getsize(os.path.join(LOCAL_LIVE, 'crnn_model.pt')) / 1e6:.1f} MB | "
      f"epoch {exp['info']['epoch']} | max output difference vs best.pt: {diff:.2e}")
assert diff < 1e-4
''')
    nb.md("""
## Check: the demo's code gives the evaluation's predictions
`N_CHECK` random IAM **test** words are read straight from `archive.zip` (no unzip) and run through the demo's `Recognizer`, then compared with Step 2's `crnn_test_predictions.csv`:
- **without the photo clean-up** the input tensor is exactly the evaluation's; predictions can still differ on a few words, because evaluation runs in padded batches (the BiLSTM also sees a little white padding) while the demo reads one word at a time. The CER should be about the same.
- **with the photo clean-up** (what the demo does to camera images): IAM scans are already clean, so this mainly shows the clean-up doesn't hurt (on the box, CER rose by ~0.3 points on 400 test words).
""")
    nb.code('''
import csv, zipfile
from src.utils.metrics import cer
N_CHECK = 200
CHECK_DIR = "/content/check_words"; os.makedirs(CHECK_DIR, exist_ok=True)
pred_csv = os.path.join(REPORT_DIR, "crnn_test_predictions.csv")
SAMPLES = []
if not os.path.exists(pred_csv):
    print("crnn_test_predictions.csv not found (run Step 2's evaluation); skipping the check.")
else:
    ev = list(csv.DictReader(open(pred_csv, newline="")))
    with zipfile.ZipFile(ARCHIVE) as z:
        members = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.endswith(".png")}
        pick = [r for r in random.Random(0).sample(ev, min(4 * N_CHECK, len(ev))) if r["word_id"] in members][:N_CHECK]
        for r in pick:
            with open(os.path.join(CHECK_DIR, r["word_id"] + ".png"), "wb") as f:
                f.write(z.read(members[r["word_id"]]))
    refs, hyp_eval, hyp_demo, hyp_clean = [], [], [], []
    for r in pick:
        g = cv2.imread(os.path.join(CHECK_DIR, r["word_id"] + ".png"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        refs.append(r["ref"]); hyp_eval.append(r["hyp"])
        hyp_demo.append(rec.read(g, clean=False)[0]); hyp_clean.append(rec.read(g)[0])
    same = np.mean([a == b for a, b in zip(hyp_demo, hyp_eval)])
    print(f"{len(refs)} IAM test words | demo == evaluation prediction on {100 * same:.1f}% of words")
    print(f"CER: evaluation {100 * cer(refs, hyp_eval):.2f}% | demo {100 * cer(refs, hyp_demo):.2f}% | "
          f"demo with photo clean-up {100 * cer(refs, hyp_clean):.2f}%")
    good = [r for r in pick if r["ref"] == r["hyp"] and len(r["ref"]) >= 4][:3]
    SAMPLES = [(os.path.join(CHECK_DIR, r["word_id"] + ".png"), r["ref"]) for r in good]
''')
    nb.md("""
## Build `live_demo` on Drive (+ `live_demo.zip`)
The folder is built on Colab's disk, tested by running `live_demo.py` **from inside the folder** (headless, on a sample word, so it only uses the bundled code), then copied to `live_demo/` in your project folder and zipped as `live_demo.zip` next to it. Existing files in `live_demo/` on Drive that aren't ours (e.g. a `webcam_test` you put there) are never deleted.
""")
    nb.code('''
files = bundle.build(LOCAL_LIVE, os.path.join(LOCAL_LIVE, "crnn_model.pt"), MODEL_INFO, SAMPLES)
if SAMPLES:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "live_demo.py", "--image", os.path.join("samples", os.path.basename(SAMPLES[0][0])),
                        "--headless", "--max-frames", "3", "--box", "0.98", "0.98"], cwd=LOCAL_LIVE, env=env,
                       capture_output=True, text=True)
    print(r.stdout[-600:] or r.stderr[-1500:])
    print("true word:", SAMPLES[0][1])
LIVE_DIR = os.path.join(PROJECT_DIR, "live_demo")
for f in files:
    os.makedirs(os.path.dirname(os.path.join(LIVE_DIR, f)), exist_ok=True)
    shutil.copy2(os.path.join(LOCAL_LIVE, f), os.path.join(LIVE_DIR, f))
LIVE_ZIP = os.path.join(PROJECT_DIR, "live_demo.zip")
shutil.copy(bundle.make_zip(LOCAL_LIVE, "/content/live_demo.zip"), LIVE_ZIP)
print(f"\\n{len(files)} files in {LIVE_DIR}")
for f in files:
    print(f"  {os.path.getsize(os.path.join(LOCAL_LIVE, f)) / 1024:9.1f} KB  {f}")
print(f"live_demo.zip: {os.path.getsize(LIVE_ZIP) / 1e6:.1f} MB  ->  {LIVE_ZIP}")
''')
    nb.md("""
## On your laptop (short version; full steps in the folder's README.md)
1. Install Python 3.12 from python.org (tick **Add python.exe to PATH**).
2. Download `live_demo.zip` from the project folder on Drive, right-click → *Extract All*.
3. In the extracted folder: address bar → `cmd` → `python -m pip install -r requirements.txt`.
4. DroidCam app on the phone + DroidCam Client on Windows, same Wi-Fi, type the phone's IP, **Start**.
5. `python live_demo.py`. WORD mode: `c` = capture + type the true word + Enter; `space` freeze; `a/d/w/s` box size; `n` next camera; `m` PAGE mode (beta); `q` quit.
6. Collect ~150 words, then drag the `webcam_test` folder into the project folder on Drive and run this notebook again.
""")
    nb.md("""
## Load Model B (augmented CRNN from Step 6)
Model A is the `crnn_model.pt` exported above (the same file as `live_demo/crnn_model.pt`). Model B is **`outputs/model_b/crnn_model_b.pt`**, saved by Step 6. Before scoring, the notebook checks that Model B has **the same architecture** as Model A (CNN channels, LSTM size/layers, dropout), the same 78-character charset and the same preprocessing, and that it matches its own training run's `config.json` (`runs/crnn_aug_*/`). Only the training data differ (camera-style augmentation), so A vs B isolates the augmentation. If Step 6 hasn't run, Model B is skipped and the table has 3 rows.
""")
    nb.code('''
from src.demo import score as S
MODEL_B_PT = os.path.join(OUTPUT_DIR, "model_b", "crnn_model_b.pt")
MODEL_B_METRICS = os.path.join(OUTPUT_DIR, "model_b", "model_b_metrics.json")
rec_b, MODEL_B_INFO = None, {}
IAM_CER = {"crnn": CRNN_METRICS["test"]["cer"]} if "cer" in CRNN_METRICS.get("test", {}) else {}
if not os.path.exists(MODEL_B_PT):
    print("Model B not found:", MODEL_B_PT, "(run Step 6 first); scoring Model A and the baselines only.")
else:
    rec_b = W.Recognizer(MODEL_B_PT)
    diffs = S.architecture_diff(rec, rec_b)
    assert not diffs, "Model B's architecture differs from Model A's: " + "; ".join(diffs)
    run_b = rec_b.info.get("run") or ""
    cfg_paths = [os.path.join(PROJECT_DIR, "runs", run_b, "config.json")] if run_b else []
    cfg_paths += sorted(glob.glob(os.path.join(PROJECT_DIR, "runs", "crnn_aug_*", "config.json")), key=os.path.getmtime)[::-1]
    cfg_path = next((p for p in cfg_paths if os.path.exists(p)), None)
    if cfg_path:
        cdiff = S.config_diff(rec_b, json.load(open(cfg_path)))
        assert not cdiff, f"crnn_model_b.pt doesn't match {cfg_path}: " + "; ".join(cdiff)
        print("Model B matches its run's config.json:", os.path.relpath(cfg_path, PROJECT_DIR))
    else:
        print("note: no runs/crnn_aug_*/config.json found; checked against Model A only")
    if os.path.exists(MODEL_B_METRICS):
        mb_test = json.load(open(MODEL_B_METRICS)).get("test", {})
        if "cer" in mb_test:
            IAM_CER["crnn_b"] = mb_test["cer"]
            print(f"Model B IAM test CER {100 * mb_test['cer']:.2f}% | WER {100 * mb_test['wer']:.2f}%")
    MODEL_B_INFO = {"file": os.path.relpath(MODEL_B_PT, PROJECT_DIR), **rec_b.info}
    print(f"Model B loaded: run {run_b or '?'}, epoch {rec_b.info.get('epoch')} | same architecture as Model A:",
          rec_b.model_cfg)
''')
    nb.md("""
# Score the webcam test set
Looks for a folder named `webcam_test` (with `labels.csv`) in your project folder (also one level down, e.g. inside `live_demo/`). Every model reads **the same cleaned crop** of each word (the same `clean_photo` as the live demo):
- **CRNN Model A** (Step 2) and **CRNN Model B** (Step 6, augmented): training transform + greedy CTC, exactly as in the demo; an extra **ablation** table reads the raw guide-box crop with Model A, without the clean-up;
- **Tesseract**: Step 3's settings (16-px white border, `--oem 1 --psm 8`);
- **EMNIST baseline**: Step 4's splitter settings and IAM character prior (chosen on IAM validation), if Step 4 has run.

Labels are compared **case-sensitively** with the same CER/WER code as before. Rows with `needs_check = 1` (PAGE-mode page captures labelled with the model's guess) are **not scored** until you fix the label and set it to 0.
""")
    nb.code('''
dirs = S.find_webcam_dirs(PROJECT_DIR)
WEBCAM_DIR, WEBCAM_ROWS, UNCHECKED = (dirs[0] if dirs else None), [], []
if WEBCAM_DIR is None:
    print("No webcam_test/ folder with labels.csv in", PROJECT_DIR)
    print("""Nothing to score yet. To make the webcam test set:
  1. Download live_demo.zip (project folder on Drive), extract it, and run:  python live_demo.py
  2. For each word: hold it in the green box, press c, type the true word, press Enter (~150 words).
  3. The captures are in the folder webcam_test next to live_demo.py (images/ + labels.csv).
  4. Drag the whole webcam_test folder into the project folder on Drive (next to archive.zip).
  5. Run this notebook again (Runtime -> Run all).""")
else:
    WEBCAM_ROWS, unreadable = S.load_rows(WEBCAM_DIR, UNCHECKED)
    print(f"{WEBCAM_DIR}: {len(WEBCAM_ROWS)} labelled words" + (f" (also found: {dirs[1:]})" if len(dirs) > 1 else ""))
    if UNCHECKED:
        print(f"  {len(UNCHECKED)} page-capture rows still have needs_check=1 (label is only a guess): not scored")
    if unreadable:
        print(f"  {len(unreadable)} unreadable image(s) skipped: {unreadable[:10]}")
    no_ink = [r["file"] for r in WEBCAM_ROWS if r["clean"] is None]
    if no_ink:
        print(f"  {len(no_ink)} capture(s) with no clear writing in the box (they score as empty predictions): {no_ink[:10]}")
    outside = sorted({c for r in WEBCAM_ROWS for c in r["label"] if c not in rec.charset.chars})
    if outside:
        print("  characters in the labels that the CRNN can't output:", outside)
    if len(WEBCAM_ROWS) < 100:
        print(f"  note: only {len(WEBCAM_ROWS)} words; the proposal plans ~150, so these numbers are noisy.")
''')
    nb.md("### Run the four models on the same words")
    nb.code('''
from src.report import webcam_report as wr
HYPS, MS, BASE_INFO = {}, {}, {}
if WEBCAM_ROWS:
    HYPS["crnn"], HYPS[wr.RAW], MS["crnn"] = S.run_crnn(WEBCAM_ROWS, rec)
    print(f"CRNN Model A done ({MS['crnn']:.1f} ms/word on the CPU)")
    if rec_b is not None:
        HYPS["crnn_b"], _, MS["crnn_b"] = S.run_crnn(WEBCAM_ROWS, rec_b, with_raw=False)
        print(f"CRNN Model B done ({MS['crnn_b']:.1f} ms/word on the CPU)")
    !apt-get -qq install -y tesseract-ocr > /dev/null
    !pip -q install pytesseract
    try:
        from src.baselines import tesseract as T
        HYPS["tesseract"], MS["tesseract"] = S.run_tesseract(WEBCAM_ROWS)
        BASE_INFO["tesseract"] = {"version": T.version(), "config": T.config(), "border_px": T.BORDER}
        print(f"Tesseract done ({MS['tesseract']:.0f} ms/word)")
    except Exception as e:
        print("Tesseract skipped:", e)
    from src.baselines import emnist as E, char_segment as cs
    em_ck = os.path.join(PROJECT_DIR, "runs", "emnist_cnn", "best.pt")
    em_json = os.path.join(REPORT_DIR, "emnist_metrics.json")
    em_labels = os.path.join(PROJECT_DIR, "emnist", E.FILES["train_y"])
    if os.path.exists(em_ck) and os.path.exists(em_json):
        st = json.load(open(em_json))["settings"]
        params = cs.SegParams(**{k: st[k] for k in cs.params_dict(cs.SegParams()) if k in st})
        prior, alpha = None, 0.0
        words_txt = OFFICIAL_WORDS_TXT if os.path.exists(OFFICIAL_WORDS_TXT) else RECOVERED_WORDS_TXT
        if st.get("iam_prior") and os.path.exists(em_labels) and os.path.exists(words_txt):
            texts = S.iam_train_texts(words_txt, "/content/aps360_htr/data/splits/aachen", rec.charset.chars)
            prior, alpha = cs.char_prior(texts, E.read_idx(em_labels)), 1.0
        clf = E.load_classifier(em_ck, torch.device("cpu"))
        HYPS["emnist"], MS["emnist"] = S.run_emnist(WEBCAM_ROWS, clf, torch.device("cpu"), params, prior, alpha)
        BASE_INFO["emnist"] = {"settings": st, "iam_prior_used": bool(alpha)}
        print(f"EMNIST baseline done ({MS['emnist']:.1f} ms/word; width factor {params.width_factor}, "
              f"IAM prior {'on' if alpha else 'off'})")
    else:
        print("EMNIST baseline skipped: run Step 4 first (needs runs/emnist_cnn/best.pt and emnist_metrics.json).")
''')
    nb.md("# Report outputs\\nSaved to `outputs/report/` (same folder as Steps 2-4): figures as PNG (300 dpi) + PDF, tables as CSV + LaTeX booktabs `.tex`.")
    nb.code('''
from IPython.display import Image, Markdown, display
from src.report.manifest import Report, future_work
from src.report.tables import csv_to_markdown
rep = Report(REPORT_DIR, step=5)
def show(paths):
    for p in paths or []:
        if p.endswith(".png"):
            display(Image(p, width=700))
show_table = lambda name: display(Markdown(csv_to_markdown(os.path.join(REPORT_DIR, name + ".csv"))))
if WEBCAM_ROWS:
    wr.save_predictions(rep, WEBCAM_ROWS, HYPS)
    info = {"folder": os.path.relpath(WEBCAM_DIR, PROJECT_DIR), "crnn_run": RUN_NAME, "model_b": MODEL_B_INFO,
            "unchecked_skipped": len(UNCHECKED),
            "no_ink": sum(r["clean"] is None for r in WEBCAM_ROWS), "baselines": BASE_INFO,
            "device": "CPU", "case_sensitive": True}
    WEBCAM_NUMS = wr.save_metrics(rep, WEBCAM_ROWS, HYPS, MS, info, iam_cer=IAM_CER)
    show_table("table_webcam_metrics")
    print("Photo clean-up ablation (Model A):")
    show_table("table_webcam_cleanup_ablation")
    print("Per-word predictions:", ", ".join(f"webcam_predictions_{k}.csv" for k in HYPS))
else:
    print("No webcam words yet: report outputs skipped (see the instructions above).")
''')
    nb.md("### Figures\\n`fig_webcam_vs_iam` (CER on IAM test vs webcam, per model), `fig_webcam_predictions` (12 random webcam words, Model A), `fig_webcam_predictions_model_b` (same words, Model B), `fig_webcam_preprocessing` (camera crop → clean-up → model input).")
    nb.code('''
if WEBCAM_ROWS:
    show(wr.save_vs_iam_figure(rep, WEBCAM_NUMS))
    show(wr.save_prediction_grid(rep, WEBCAM_ROWS, HYPS["crnn"]))
    if "crnn_b" in HYPS:
        show(wr.save_prediction_grid(rep, WEBCAM_ROWS, HYPS["crnn_b"], name="fig_webcam_predictions_model_b",
                                     model="CRNN Model B"))
    show(wr.save_preprocessing_figure(rep, WEBCAM_ROWS, rec))
''')
    nb.md("### Summary and list of files")
    nb.code('''
summary = rep.write_summary(future_work(rep))
display(Markdown(open(summary).read()))
print("Saved report files:")
rep.list_files();
''')
    nb.md("""
## Notes for the report
- **Domain gap:** IAM is scanned forms; the webcam set is phone-camera photos (blur, uneven light, JPEG, different pens and writers). Expect a higher CER than on IAM for every model; `table_webcam_metrics` puts both side by side.
- **Same input for every model:** all four read the same cleaned crop, so the comparison is fair. `table_webcam_cleanup_ablation` (Model A, raw crop) shows how much the clean-up itself is worth.
- **Model A vs Model B:** same architecture, charset and preprocessing (checked above); only the training augmentation differs, so the webcam CER gap between the two rows is the effect of augmentation. On clean IAM test, B is already better (8.77% vs 10.75% CER).
- **Small test set:** ~150 words from few writers, so ±several points of CER is noise. Say how many words and writers.
- **Preprocessing = training:** the demo and this notebook use the training transform unchanged (32 px high, aspect kept, [-1, 1]); only the photo clean-up before it is new.
- **PAGE mode is a beta extra:** it is not used for any number in the report.
- **Per-word predictions:** `webcam_predictions.csv` (all models side by side) and `webcam_predictions_<model>.csv` (one per model: file, label, prediction, correct, cer).
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step5_Webcam.ipynb")


# ------------------------------------------------------------------ step 6 (Model B: augmentation)

def build_step6():
    nb = NB()
    nb.md("""
# APS360 Handwriting Recognition: Step 6 (Model B: CRNN + camera-style augmentation)

**Model A** (Step 2) is the CRNN trained on clean IAM scans. **Model B** is the *same* CRNN with the *same* settings, trained on the same IAM words, except that every training word is randomly distorted to look like a phone/webcam photo. Comparing A and B shows what augmentation does.

### What is data augmentation, and why?
A neural network only learns what it sees. IAM words are clean, flat, evenly lit **scans**. A word seen by a webcam or phone looks different:

| Camera problem | What the augmentation does |
|---|---|
| paper tilted, or seen at an angle | small **rotation**, **perspective warp** and slant (**shear**) |
| shaky hand or moving camera | **motion blur** (or a little focus blur) |
| desk lamp from one side, hand or phone shadow | **uneven lighting** and a soft **shadow** band |
| faded pen, grey paper, cheap camera | lower **contrast**, brightness shift, sensor **noise** |
| phone / webcam video compression | **JPEG** compression |

Each time a training word is used, a random mix of these is applied (each with some probability, at a random strength), so the model almost never sees exactly the same picture twice. The **label stays the same**, so the model learns to read *through* the camera problems. This usually costs a little accuracy on clean scans (the task got harder) but should make the model more robust on real camera images.

### What stays exactly the same as Notebook 2
- the IAM word images, labels (`words_recovered.txt`), filters, skip list / bad-image handling and the **Aachen train/val/test splits**;
- the CRNN architecture and the **hyperparameters of Notebook 2's best run**, read from `outputs/sweep_results.csv` exactly like Step 2 (the sweep is **not** re-run; defaults if there is no CSV);
- `EPOCHS = 100`, early stopping (patience 8), LR halving after 3 epochs without improvement, AdamW, mixed precision, resume from `last.pt`.

### Rules
- Augmentation is applied to the **training split only**. Validation and test are the clean IAM words, so A and B are compared on identical data.
- The webcam test set (`webcam_test/`) is **not used or scored here** (that comes later).
- Model B writes only to its own folders: **`runs/crnn_aug_<settings>/`** (checkpoints) and **`outputs/model_b/`** (results, figures). Notebook 2's `runs/crnn_*` and `outputs/report/` are only *read* (for the A vs B table), never written.

| Section | What happens |
|---|---|
| Setup | GPU check, Drive, code, data (same as Step 2) |
| Augmentation preview | example augmented words, saved as report figures |
| Train Model B | same settings as Model A's best run; re-run to resume |
| Evaluate | CER / WER on **clean** IAM val and test, A vs B table |

**Before you run:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all. Run Step 1 once before (so `words_recovered.txt` exists) and ideally Step 2 (so the sweep winner and Model A's numbers exist).

**Time (estimate, not measured on a T4):** ~10 min setup (unzip + load), then roughly **1–1.5 min per epoch** (a bit slower than Step 2 because every word is distorted on the CPU). With early stopping that's usually **~1–2 hours**; the worst case (all 100 epochs) is about 2–2.5 hours. If Colab disconnects, Run all again: it resumes from the last finished epoch.
""")
    nb.md("## Check the GPU\nYou should see a Tesla T4. If it says CPU, change the runtime type: training on CPU would take many hours.")
    nb.code('''
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU, then Runtime -> Restart and run all.")
''')
    mount(nb)
    paths(nb, extra='MODEL_B_DIR = os.path.join(OUTPUT_DIR, "model_b")          # Model B results (separate from outputs/report)\n'
                    'os.makedirs(MODEL_B_DIR, exist_ok=True)\n'
                    'print("MODEL_B_DIR:", MODEL_B_DIR)\n')
    _GROUPS["current"] = [
        ("### Camera-style augmentation (new for Model B)\n`CameraAugment` applies a random mix of the distortions to a uint8 grayscale word image (same size out). `AugmentedWords` wraps the **training** dataset only: it keeps the training images in RAM at 64 px high, distorts them, then resizes to 32 px with the same preprocessing as Model A. The random choices come from (seed, epoch, word), so the run is reproducible.",
         ["src/data/augment.py"]),
        ("### Model B report outputs\nFigures and tables saved to `outputs/model_b/`.", ["src/report/model_b_report.py"]),
        ("### Training loop (same as Step 2)\nIdentical to Step 2, plus one line: each epoch it tells the augmented dataset the epoch number, so every epoch gets new random distortions.",
         ["src/training/engine.py"]),
    ]
    readable = [f for _, fs in _GROUPS["current"] for f in fs]
    write_code(nb, readable,
               "The new Model B code is shown below as `%%writefile` cells. Everything else (data pipeline, CRNN, sweep helpers, report helpers) is the same code as Steps 1-5 and comes from the embedded zip.",
               """| File | What it does |
|---|---|
| `src/data/augment.py` | the camera-style distortions (`CameraAugment`, `AugConfig`) and the training-only wrapper `AugmentedWords` |
| `src/report/model_b_report.py` | augmentation example figures, training curves, metrics and the Model A vs B table, all in `outputs/model_b/` |
| `src/training/engine.py` | `fit()` from Step 2 (AdamW, mixed precision, LR halving, early stopping, resume) |""")
    install(nb); unzip_and_setup(nb); recover(nb)
    tests(nb, "tests/test_augment.py tests/test_training.py tests/test_model.py tests/test_data.py",
          "The augmentation tests (same shape and type out, only applied in training mode, same images for the same seed, val/test stay clean) plus the model, data and training tests from Step 2.")
    nb.md("""
## Load the data (same as Step 2)
Same `load_data()` as Step 2: same filters, skip list (`a01-117-05-02`, `r06-022-03-05`, ...), Aachen splits and bad-image handling. Validation is preloaded at 32 px exactly like Step 2.

The **training** words are then wrapped by `AugmentedWords`, which preloads them at 64 px (twice the model's height) so the distortions look natural, and distorts them on the fly. That takes a few minutes and about 1 GB of RAM. Any image that can't be decoded is dropped and counted as `bad_or_missing_image`, like in Step 2.
""")
    nb.code('''
from src.training.data import load_data, add_dropped
from src.data.augment import AugConfig, AugmentedWords, CameraAugment
AUG_SEED = 42
AUG_CFG = AugConfig()          # probabilities and strengths of each distortion (see the code above)
datasets, charset, stats = load_data(DATA_ROOT, splits=("val",), preload=True, check_images=False)
import time; t = time.time()
train_b = AugmentedWords(datasets["train"], CameraAugment(AUG_CFG), seed=AUG_SEED)
add_dropped(stats, {"train": train_b})
print(f"training words (augmented): {len(train_b)} | val words (clean): {len(datasets['val'])} | {time.time() - t:.0f}s")
print({k: stats[k] for k in ("kept", "segmentation_err", "bad_or_missing_image", "malformed_line", "split_method")})
assert not isinstance(datasets["val"], AugmentedWords) and datasets["val"].transform.augment is None
''')
    nb.md("""
## Augmentation preview (figures for the report)
- `fig_model_b_augmentation_examples`: four training words, clean and with four random augmentations (exactly what Model B trains on);
- `fig_model_b_augmentation_types`: one word with each distortion on its own.

Saved to `outputs/model_b/` as PNG (300 dpi) + PDF. If the words look unreadable to *you*, the augmentation is too strong: lower the numbers in `AugConfig` and re-run (that changes the training, so use it before training starts).
""")
    nb.code('''
from IPython.display import Image, display
from src.report import model_b_report as mb
for p in mb.save_augmentation_examples(MODEL_B_DIR, train_b, seed=AUG_SEED):
    if p.endswith(".png"):
        display(Image(p, width=700))
''')
    nb.md("""
## Train Model B
**Settings:** the same choice as Step 2's Train cell: the best row of `outputs/sweep_results.csv` (lowest val CER, ties by val WER), or the defaults (lr 1e-3, hidden 256, dropout 0.3, batch 64, 2 LSTM layers) if there's no usable CSV. The sweep is **not** re-run.

- `EPOCHS = 100`, early stopping after 8 epochs without a val-CER improvement, LR halves after 3, AdamW, mixed precision: all as in Step 2.
- Checkpoints go to **`runs/crnn_aug_<settings>/`** (e.g. `runs/crnn_aug_lr0.001_h256_d0.3_b32_L2`): `best.pt`, `last.pt`, `history.csv`, `config.json`, `augmentation.json`. That name can never be the same as Model A's `runs/crnn_<settings>/`.
- **If Colab disconnects:** Run all again; this cell resumes from `last.pt`. It refuses to resume if the settings or the augmentation changed (then it tells you why).
- Validation each epoch uses the **clean** val words.
- Two DataLoader workers distort the next batches on the CPU while the GPU trains.
""")
    nb.code('''
import json
from src.training.engine import fit
from src.training.sweep import choose_settings, make_configs, check_run_dir, run_name
EPOCHS = 100
SWEEP_CSV = os.path.join(OUTPUT_DIR, "sweep_results.csv")      # read only (Step 2's sweep)
SETTINGS, SETTINGS_INFO = choose_settings(SWEEP_CSV)
print(f"Settings from: {SETTINGS_INFO['source'].upper()} - {SETTINGS_INFO['reason']}")
for k, v in SETTINGS.items():
    print(f"  {k:12s} = {v:g}" if isinstance(v, float) else f"  {k:12s} = {v}")

RUN_NAME_A = "crnn_" + run_name(SETTINGS)                    # Model A (Step 2), only read
RUN_NAME   = "crnn_aug_" + run_name(SETTINGS)                # Model B
RUN_DIR    = os.path.join(PROJECT_DIR, "runs", RUN_NAME)
assert RUN_NAME != RUN_NAME_A and os.path.basename(RUN_DIR).startswith("crnn_aug_")
os.makedirs(RUN_DIR, exist_ok=True)
model_cfg, train_cfg = make_configs(SETTINGS, epochs=EPOCHS, patience=8, num_workers=2, out_dir=RUN_DIR)
print("RUN_DIR:", RUN_DIR, "|", check_run_dir(RUN_DIR, model_cfg, train_cfg))

aug_info = {"seed": AUG_SEED, "config": AUG_CFG.to_dict(), "work_height": 64}
aug_json = os.path.join(RUN_DIR, "augmentation.json")
if os.path.exists(aug_json) and json.load(open(aug_json)) != aug_info:
    raise RuntimeError(f"{RUN_DIR} was trained with a different augmentation, so it can't be resumed.\\n"
                       f"saved: {json.load(open(aug_json))}\\nnow:   {aug_info}\\n"
                       "Put AugConfig back as it was, or delete that run folder to start over.")
json.dump(aug_info, open(aug_json, "w"), indent=2)
json.dump({"settings": SETTINGS, **SETTINGS_INFO}, open(os.path.join(RUN_DIR, "settings_choice.json"), "w"),
          indent=2, default=str)
result = fit({"train": train_b, "val": datasets["val"]}, charset, model_cfg, train_cfg, resume=True)
print("best val CER (clean val):", result["best_val_cer"])
''')
    nb.md("## Training curves\nSaved as `fig_model_b_training_curves`, `fig_model_b_loss_curve` and `fig_model_b_val_cer_wer_curve` in `outputs/model_b/`. Train loss is on *augmented* words and val loss on *clean* words, so train loss can sit above val loss: that's normal here.")
    nb.code('''
for p in mb.save_curves(MODEL_B_DIR, os.path.join(RUN_DIR, "history.csv")):
    if p.endswith(".png"):
        display(Image(p, width=700))
''')
    nb.md("""
## Evaluate Model B on clean IAM val and test
`best.pt` is reloaded and greedy-decoded on the **clean** validation and test words (unseen writers, Aachen split), exactly like Step 2. If Model A's `runs/crnn_<settings>/metrics.json` exists (from Step 2), the **Model A vs Model B** table is made too.

Saved to `outputs/model_b/`: `model_b_metrics.json`, `table_model_b_metrics` (CSV + LaTeX), `table_model_a_vs_b` (CSV + LaTeX), `model_b_val_predictions.csv`, `model_b_test_predictions.csv`, and `crnn_model_b.pt` (small CPU copy of the weights, like the live demo's `crnn_model.pt`, for scoring the webcam set later).
""")
    nb.code('''
import torch
from IPython.display import Markdown
from src.training.engine import evaluate, get_device, load_model, make_loader
from src.report.predict import predict_dataset, save_predictions
from src.report.tables import csv_to_markdown
from src.demo.webcam import export_model
device = get_device()
model, cs, ck = load_model(os.path.join(RUN_DIR, "best.pt"), device)
print(f"Model B best epoch {ck['epoch']}, val CER {ck['best_cer']:.4f}")
test_sets, _, _ = load_data(DATA_ROOT, splits=("test",), preload=True, check_images=False)
add_dropped(stats, test_sets)
eval_sets = {"val": datasets["val"], "test": test_sets["test"]}     # clean, never augmented
results, preds = {}, {}
for split, ds in eval_sets.items():
    loader, _ = make_loader(ds, 128, shuffle=False)
    results[split] = evaluate(model, loader, cs, device, amp=device.type == "cuda")
    preds[split] = predict_dataset(model, ds, cs, device, batch_size=128, amp=device.type == "cuda")
    save_predictions(preds[split], os.path.join(MODEL_B_DIR, f"model_b_{split}_predictions.csv"))
    r = results[split]
    print(f"{split:4s}: {r['n']} words | CER {r['cer']:.4f} | WER {r['wer']:.4f}")
json.dump({s: {k: results[s][k] for k in ("cer", "wer", "n", "ms_per_word")} for s in results},
          open(os.path.join(RUN_DIR, "metrics.json"), "w"), indent=2)

model_a = mb.load_model_a_metrics(os.path.join(PROJECT_DIR, "runs", RUN_NAME_A))
print("Model A metrics:", "found in runs/" + RUN_NAME_A if model_a else "not found (run Step 2's Evaluate cell); no A vs B table")
nums, paths = mb.save_metrics(MODEL_B_DIR, preds, results, SETTINGS, RUN_DIR, aug_info, model_a)
for p in paths:
    if p.endswith(".csv"):
        display(Markdown(csv_to_markdown(p)))
export_model(os.path.join(RUN_DIR, "best.pt"), os.path.join(MODEL_B_DIR, "crnn_model_b.pt"),
             extra={"model": "B (camera-style augmentation)", "run": RUN_NAME})
print("saved crnn_model_b.pt")
''')
    nb.code('''
print("Files in outputs/model_b:")
for f in sorted(os.listdir(MODEL_B_DIR)):
    print(f"  {os.path.getsize(os.path.join(MODEL_B_DIR, f)) / 1024:8.1f} KB  {f}")
''')
    nb.md("""
## Next
- **Webcam scoring is in Step 5:** once `webcam_test/` is collected, re-run Step 5; it scores Model A (`live_demo/crnn_model.pt`), Model B (`outputs/model_b/crnn_model_b.pt`, saved here), Tesseract and EMNIST on the same words. That's where augmentation should help most.
- For the report: `fig_model_b_augmentation_examples` (what augmentation looks like), `table_model_a_vs_b` (clean IAM: what it costs or gains), training curves.
""")
    nb.save(R / "notebooks" / "APS360_HTR_Step6_ModelB_Augment.ipynb")


if __name__ == "__main__":
    build_step1()
    build_step2()
    build_step3()
    build_step4()
    build_step5()
    build_step6()

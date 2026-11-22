"""Central settings for the project.

Every script reads its defaults from here, so a change in one place applies
everywhere. Command-line flags in the scripts can override most of them.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = the folder that contains src/ (works on a laptop or in Colab).
ROOT = Path(__file__).resolve().parents[1]

# Where the IAM data lives. Set the HTR_DATA_ROOT environment variable (or pass
# --data-root to the scripts) to point somewhere else, e.g. in Colab:
#   os.environ["HTR_DATA_ROOT"] = "/content/iam"
#   or "/content/drive/MyDrive/APS360/iam" (works, but reading from Drive is slow)
DATA_ROOT = Path(os.environ.get("HTR_DATA_ROOT", ROOT / "data" / "iam"))


@dataclass
class DataConfig:
    # Where IAM lives. Expected layout (same for the fake stand-in data):
    #   <iam_root>/ascii/words.txt
    #   <iam_root>/ascii/forms.txt            (optional, gives writer IDs)
    #   <iam_root>/words/a01/a01-000u/a01-000u-00-00.png
    iam_root: Path = DATA_ROOT

    # Optional split lists (one ID per line: form, line or word ID).
    # Defaults to the Aachen splits (form IDs) that ship in data/splits/aachen.
    # Set split_dir=None to always use the automatic writer-disjoint split.
    split_dir: Path | None = ROOT / "data" / "splits" / "aachen"
    split_files: dict = field(default_factory=lambda: {
        "train": "train.uttlist",
        "val": "validation.uttlist",
        "test": "test.uttlist",
    })

    # Form -> writer table, used when the download has no ascii/forms.txt (e.g. Kaggle).
    # See data/iam_form_writers_SOURCE.md for where it comes from. None = don't use it.
    form_writers_csv: Path | None = ROOT / "data" / "iam_form_writers.csv"

    # Fallback split (used when split files are missing): fractions of items
    # per split, grouped so that no writer appears in two splits.
    split_fractions: tuple = (0.8, 0.1, 0.1)   # train, val, test
    split_seed: int = 42

    # Filtering
    drop_err: bool = True        # drop words whose segmentation status is "err"
    max_label_len: int = 32      # drop longer words (also keeps CTC happy)
    check_images: bool = True    # open every image once to drop broken/empty files

    # Image preprocessing
    img_height: int = 32         # brief: resize to 32 px high
    min_width: int = 16          # very short words (e.g. ".") get at least this
    max_width: int = 512         # very long words get squeezed to this

    # Loader
    batch_size: int = 32
    num_workers: int = 0         # 0 is safest on Windows/macOS; raise on Linux


@dataclass
class ModelConfig:
    """CRNN size. Saved inside every checkpoint so evaluate.py can rebuild it."""
    cnn_channels: tuple = (64, 128, 256, 256, 512)  # the 5 conv blocks
    lstm_hidden: int = 256       # per direction (BiLSTM output = 2x this)
    lstm_layers: int = 2
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3             # peak learning rate (AdamW)
    weight_decay: float = 1e-4
    grad_clip: float = 5.0       # LSTMs + CTC can produce the odd huge gradient
    patience: int = 8            # stop if val CER hasn't improved for this many epochs
    lr_patience: int = 3         # halve the LR after this many epochs without improvement
    amp: bool = True             # mixed precision on GPU (faster on a T4)
    num_workers: int = 2         # Colab has 2 CPU cores
    preload: bool = True         # keep resized images in RAM (much faster epochs)
    train_frac: float = 1.0      # <1 trains on a random subset (quick sweeps)
    max_batches: int | None = None  # cap batches per epoch (smoke tests)
    seed: int = 42
    out_dir: Path = ROOT / "outputs" / "runs" / "crnn"

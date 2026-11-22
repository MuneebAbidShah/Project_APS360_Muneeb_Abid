"""IAM words: parse words.txt, filter, split by writer, and load as a Dataset.

words.txt line format (lines starting with '#' are comments):

    a01-000u-00-00 ok 154 408 768 27 51 AT A
    |              |  |   |   |   |  |  |  +-- transcription (the label)
    |              |  |   +---+---+--+  +----- grammatical tag
    |              |  |   bounding box x y w h
    |              |  +-- graylevel used to binarize
    |              +----- segmentation status: ok / err
    +-------------------- word ID = form-line-word

Image path for that ID:  words/a01/a01-000u/a01-000u-00-00.png
"""
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import DataConfig
from src.data.charset import Charset
from src.data.transforms import PAD_VALUE, Preprocess, to_tensor_normalized

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class WordItem:
    word_id: str   # a01-000u-00-00
    form_id: str   # a01-000u
    writer: str | None   # writer ID (forms.txt or the bundled table), or None if unknown
    path: Path
    text: str


# ---------------------------------------------------------------- parsing

def word_id_parts(word_id: str):
    """'a01-000u-00-00' -> form 'a01-000u', line 'a01-000u-00'."""
    parts = word_id.split("-")
    return "-".join(parts[:2]), "-".join(parts[:3])


def image_path(root: Path, word_id: str) -> Path:
    form_id, _ = word_id_parts(word_id)
    return root / "words" / form_id.split("-")[0] / form_id / f"{word_id}.png"


def parse_words_txt(path: Path, malformed: list | None = None):
    """Yield (word_id, status, text) for every non-comment line.

    Lines with fewer than 9 fields can't be used (no label). They're skipped
    and appended to `malformed` so the caller can report them. In practice
    this happens when a words.txt file has been cut off part-way through.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(" ")
            if len(fields) < 9:
                log.warning("skipping malformed line: %r", line)
                if malformed is not None:
                    malformed.append(line)
                continue
            # The label is everything after the 8 fixed fields.
            yield fields[0], fields[1], " ".join(fields[8:])


def load_writers_csv(csv_path: Path) -> dict:
    """data/iam_form_writers.csv -> {form_id: writer_id}."""
    writers = {}
    with open(csv_path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            if line.strip():
                form, writer = line.strip().split(",")
                writers[form] = writer
    return writers


def find_writers(cfg: DataConfig):
    """Return ({form_id: writer_id} or None, description of the source)."""
    forms_txt = Path(cfg.iam_root) / "ascii" / "forms.txt"
    if forms_txt.exists():
        return load_writers(forms_txt), "forms.txt"
    if cfg.form_writers_csv and Path(cfg.form_writers_csv).exists():
        return load_writers_csv(Path(cfg.form_writers_csv)), "bundled iam_form_writers.csv"
    return None, "unavailable"


def load_writers(forms_txt: Path) -> dict:
    """forms.txt -> {form_id: writer_id}. Line format: 'a01-000u 000 2 prt ...'."""
    writers = {}
    with open(forms_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            writers[fields[0]] = fields[1]
    return writers


# Word images that are known to be broken (empty/corrupt PNG) in the IAM release.
# They are always dropped and counted as "bad_or_missing_image".
KNOWN_BAD_WORD_IDS = frozenset({
    "a01-117-05-02",
    "r06-022-03-05",
})


def file_ok(path: Path) -> bool:
    """Cheap check (no decoding): the file exists and is not empty."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def image_ok(path: Path) -> bool:
    """False if the file is missing, empty, or not a readable image."""
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.verify()  # cheap integrity check, doesn't decode all pixels
        with Image.open(path) as im:
            return im.size[0] > 0 and im.size[1] > 0
    except Exception:
        return False


def load_items(cfg: DataConfig, charset: Charset):
    """Read words.txt and apply all filters.

    Returns (items, drop_counts). drop_counts records how many words each
    filter removed, in the order the filters are applied; the report needs these.
    """
    root = Path(cfg.iam_root)
    words_txt = root / "ascii" / "words.txt"
    if not words_txt.exists():
        raise FileNotFoundError(f"{words_txt} not found. See README for the expected layout.")

    writers, writer_source = find_writers(cfg)
    if writers is None:
        log.warning("No writer IDs (no forms.txt or bundled table): writer checks are skipped")

    # Start every filter at 0 so the report table always lists all of them.
    drops = Counter({k: 0 for k in ("segmentation_err", "empty_label", "chars_outside_charset",
                                    "too_long", "bad_or_missing_image")})
    candidates, total = [], 0
    malformed = []
    for word_id, status, text in parse_words_txt(words_txt, malformed):
        total += 1
        if cfg.drop_err and status == "err":
            drops["segmentation_err"] += 1
            continue
        if not text:
            drops["empty_label"] += 1
            continue
        if not charset.is_valid(text):
            drops["chars_outside_charset"] += 1
            continue
        if len(text) > cfg.max_label_len:
            drops["too_long"] += 1
            continue
        form_id, _ = word_id_parts(word_id)
        writer = writers.get(form_id) if writers else None
        candidates.append(WordItem(word_id, form_id, writer, image_path(root, word_id), text))

    # Last filter: bad or missing images.
    #  - known broken files are always dropped;
    #  - check_images=True opens every image (thorough, ~1-2 min on all of IAM);
    #  - check_images=False only checks that each file exists and isn't empty
    #    (a few seconds). Anything still unreadable is dropped later, when the
    #    images are loaded (IAMWords.preload / drop_unreadable), never a crash.
    # Threads help because the time is spent waiting on the disk, not the CPU.
    known = [it for it in candidates if it.word_id in KNOWN_BAD_WORD_IDS]
    candidates = [it for it in candidates if it.word_id not in KNOWN_BAD_WORD_IDS]
    with ThreadPoolExecutor(max_workers=16) as pool:
        ok = list(pool.map(image_ok if cfg.check_images else file_ok, (it.path for it in candidates)))
    items = [it for it, good in zip(candidates, ok) if good]
    bad_ids = sorted([it.word_id for it in known] + [it.word_id for it, good in zip(candidates, ok) if not good])
    drops["bad_or_missing_image"] += len(bad_ids)
    if bad_ids:
        print(f"dropped {len(bad_ids)} bad or missing word image(s): {', '.join(bad_ids[:10])}"
              + (" ..." if len(bad_ids) > 10 else ""))

    drops["malformed_line"] = len(malformed)
    drops["total_lines"] = total
    drops["writer_source"] = writer_source
    drops["kept"] = len(items)
    drops["bad_image_ids"] = bad_ids
    return items, dict(drops)


# ---------------------------------------------------------------- splitting

def split_from_files(items, split_dir: Path, split_files: dict):
    """Assign items using ID lists (form, line or word IDs, one per line).

    Works with the Aachen splits (form IDs like 'a01-000u'). Items whose form
    isn't in any list are left out and counted as 'not_in_any_split'.
    """
    id_to_split = {}
    for split, fname in split_files.items():
        for line in open(Path(split_dir) / fname, encoding="utf-8"):
            if line.strip():
                id_to_split[line.strip()] = split

    out = {s: [] for s in SPLITS}
    missing = 0
    for it in items:
        form_id, line_id = word_id_parts(it.word_id)
        split = id_to_split.get(it.word_id) or id_to_split.get(line_id) or id_to_split.get(form_id)
        if split is None:
            missing += 1
        else:
            out[split].append(it)
    return out, missing


def writer_disjoint_split(items, fractions=(0.8, 0.1, 0.1), seed=42):
    """Deterministic split where each writer goes to exactly one split.

    If writer IDs are unknown, whole forms are grouped by form prefix (e.g. 'a01')
    instead. That keeps each form together but can't guarantee unseen writers.

    Writers are sorted, shuffled with a fixed seed, then handed to train until
    it holds ~80% of the words, then val, then test. Same data + same seed
    -> same split every time.
    """
    by_writer = defaultdict(list)
    for it in items:
        by_writer[it.writer or "prefix:" + it.form_id.split("-")[0]].append(it)
    writers = sorted(by_writer)
    random.Random(seed).shuffle(writers)

    n = len(items)
    cut_train = fractions[0] * n
    cut_val = (fractions[0] + fractions[1]) * n
    out = {s: [] for s in SPLITS}
    count = 0
    for w in writers:
        split = "train" if count < cut_train else "val" if count < cut_val else "test"
        out[split].extend(by_writer[w])
        count += len(by_writer[w])
    return out


def build_splits(cfg: DataConfig, charset: Charset):
    """Load, filter and split IAM. Returns (splits_dict, stats_dict).

    Uses the split files if they all exist and match at least one word;
    otherwise falls back to the automatic writer-disjoint split.
    """
    items, stats = load_items(cfg, charset)
    splits = None

    sd = Path(cfg.split_dir) if cfg.split_dir else None
    if sd and all((sd / f).exists() for f in cfg.split_files.values()):
        splits, missing = split_from_files(items, sd, cfg.split_files)
        if all(splits[s] for s in SPLITS):
            stats["split_method"] = f"split files in {sd}"
            stats["not_in_any_split"] = missing
        else:
            log.warning("Split files left a split empty (IDs don't match this data); "
                        "using writer-disjoint split instead")
            splits = None

    if splits is None:
        splits = writer_disjoint_split(items, cfg.split_fractions, cfg.split_seed)
        stats["split_method"] = f"writer-disjoint (seed={cfg.split_seed})"

    log.info("IAM filtering: %s", stats)
    for s in SPLITS:
        log.info("%s: %d words", s, len(splits[s]))
    return splits, stats


# ---------------------------------------------------------------- Dataset

class IAMWords(Dataset):
    """Returns (image (1,32,W), encoded target (L,), target length, text)."""

    def __init__(self, items, charset: Charset, transform: Preprocess):
        self.items, self.charset, self.transform = items, charset, transform
        self._cache = None   # filled by preload(): resized uint8 images
        self.dropped = []    # word IDs removed because their image couldn't be read
        self.skipped = set() # word IDs skipped on the fly in __getitem__ (not preloaded)

    def __len__(self):
        return len(self.items)

    def _drop(self, bad_idx, reason):
        """Remove items whose image could not be read (and remember them)."""
        if not bad_idx:
            return
        bad = set(bad_idx)
        ids = [self.items[i].word_id for i in sorted(bad)]
        self.dropped += ids
        self.items = [it for i, it in enumerate(self.items) if i not in bad]
        if self._cache is not None:
            self._cache = [a for i, a in enumerate(self._cache) if i not in bad]
        msg = (f"dropped {len(ids)} unreadable image(s) while {reason}: {', '.join(ids[:10])}"
               + (" ..." if len(ids) > 10 else ""))
        print(msg)

    def _load_resized(self, it):
        """Resized uint8 image, or None if the file can't be decoded."""
        try:
            with Image.open(it.path) as im:
                img = im.convert("L")
            if img.size[0] == 0 or img.size[1] == 0:
                return None
            return np.asarray(self.transform.resize(img), dtype=np.uint8)
        except Exception:
            return None

    def preload(self, num_threads: int = 8):
        """Load and resize every image once and keep it in RAM (uint8).

        Decoding PNGs is the slowest part of training on Colab (only 2 CPU
        cores), so this makes every epoch after the first much faster.
        ~100k IAM words at 32 px high take roughly 300 MB.
        Images that can't be decoded are dropped (listed in self.dropped).
        Not used when an augmentation is set, since that must run on the
        original image each time (then use drop_unreadable()).
        """
        if self.transform.augment is not None:
            log.warning("preload skipped: augmentation is on")
            return self.drop_unreadable(num_threads)
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            cache = list(pool.map(self._load_resized, self.items))
        self._cache = cache
        self._drop([i for i, a in enumerate(cache) if a is None], "preloading")
        return self

    def drop_unreadable(self, num_threads: int = 8):
        """Decode every image once (without keeping it) and drop the unreadable ones."""
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            ok = list(pool.map(lambda it: image_ok(it.path), self.items))
        self._drop([i for i, good in enumerate(ok) if not good], "checking images")
        return self

    def widths(self, num_threads: int = 8) -> list[int]:
        """Width of each image after resizing (used to batch similar widths).
        An unreadable file gets the minimum width here; __getitem__ skips it."""
        if self._cache is not None:
            return [a.shape[1] for a in self._cache]
        def width(it):
            try:
                with Image.open(it.path) as im:   # only reads the header
                    w, h = im.size
                return self.transform.resized_width(w, h)
            except Exception:
                return self.transform.min_width
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            return list(pool.map(width, self.items))

    def __getitem__(self, idx):
        """Returns None for an unreadable image (collate_fn leaves it out of the batch)."""
        it = self.items[idx]
        if self._cache is not None:
            x = to_tensor_normalized(Image.fromarray(self._cache[idx]))
        else:
            try:
                with Image.open(it.path) as im:
                    img = im.convert("L")  # grayscale
                x = self.transform(img)
            except Exception as e:
                if it.word_id not in self.skipped:
                    self.skipped.add(it.word_id)
                    msg = f"skipping unreadable image {it.word_id} ({type(e).__name__})"
                    print(msg)
                return None
        target = torch.tensor(self.charset.encode(it.text), dtype=torch.long)
        return x, target, len(target), it.text


def collate_fn(batch):
    """Pad images on the right to the widest one; pack targets for CTC.

    Returns a dict:
      images         (B, 1, H, W_max)  padded with paper-white
      widths         (B,)  original widths, to compute CTC input lengths later
      targets        (sum of lengths,) all labels concatenated (nn.CTCLoss format)
      target_lengths (B,)
      texts          list of B strings
    """
    batch = [b for b in batch if b is not None]     # unreadable images were returned as None
    if not batch:
        return None
    images, targets, lengths, texts = zip(*batch)
    h = images[0].shape[1]
    w_max = max(im.shape[2] for im in images)
    padded = torch.full((len(images), 1, h, w_max), PAD_VALUE)
    for i, im in enumerate(images):
        padded[i, :, :, : im.shape[2]] = im
    return {
        "images": padded,
        "widths": torch.tensor([im.shape[2] for im in images], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(lengths, dtype=torch.long),
        "texts": list(texts),
    }


def make_datasets(cfg: DataConfig | None = None, charset: Charset | None = None, train_augment=None):
    """One-call helper: returns ({'train': ds, 'val': ds, 'test': ds}, stats)."""
    cfg = cfg or DataConfig()
    charset = charset or Charset()
    splits, stats = build_splits(cfg, charset)
    base = dict(height=cfg.img_height, min_width=cfg.min_width, max_width=cfg.max_width)
    ds = {
        s: IAMWords(splits[s], charset,
                    Preprocess(**base, augment=train_augment if s == "train" else None))
        for s in SPLITS
    }
    return ds, stats

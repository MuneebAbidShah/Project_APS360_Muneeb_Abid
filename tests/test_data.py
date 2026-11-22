import torch
from PIL import Image

from src.data.charset import Charset
from src.data.iam import WordItem, collate_fn, parse_words_txt, writer_disjoint_split, IAMWords
from src.data.transforms import PAD_VALUE, Preprocess


def test_collate_shapes_and_padding():
    cs = Charset()
    batch = [
        (torch.zeros(1, 32, 40), torch.tensor(cs.encode("cat")), 3, "cat"),
        (torch.zeros(1, 32, 100), torch.tensor(cs.encode("house")), 5, "house"),
    ]
    out = collate_fn(batch)
    assert out["images"].shape == (2, 1, 32, 100)             # padded to widest
    assert torch.all(out["images"][0, :, :, 40:] == PAD_VALUE)  # padding is paper-white
    assert out["widths"].tolist() == [40, 100]
    assert out["targets"].shape == (8,)                        # 3 + 5 concatenated
    assert out["target_lengths"].tolist() == [3, 5]
    assert cs.decode(out["targets"][:3].tolist()) == "cat"


def test_preprocess_height_and_width_limits():
    pre = Preprocess(height=32, min_width=16, max_width=100)
    assert pre(Image.new("L", (200, 64), 255)).shape == (1, 32, 100)  # 100 wide after resize
    assert pre(Image.new("L", (5, 64), 255)).shape == (1, 32, 16)     # clamped up to min
    assert pre(Image.new("L", (900, 64), 255)).shape == (1, 32, 100)  # clamped down to max
    x = pre(Image.new("L", (64, 32), 255))
    assert torch.allclose(x, torch.ones_like(x))                      # white -> +1


def test_parse_words_txt(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("# comment\na01-000u-00-00 ok 154 408 768 27 51 AT A\n"
                 "a01-000u-00-01 err 154 507 766 213 48 NN MOVE\n")
    assert list(parse_words_txt(f)) == [("a01-000u-00-00", "ok", "A"),
                                        ("a01-000u-00-01", "err", "MOVE")]


def test_writer_disjoint_split_is_deterministic():
    items = [WordItem(f"f{w}-000-00-{i:02d}", f"f{w}-000", str(w), None, "a")
             for w in range(20) for i in range(5)]
    a = writer_disjoint_split(items, seed=1)
    b = writer_disjoint_split(items, seed=1)
    assert {k: [i.word_id for i in v] for k, v in a.items()} == \
           {k: [i.word_id for i in v] for k, v in b.items()}
    writers = [{i.writer for i in a[s]} for s in ("train", "val", "test")]
    assert not (writers[0] & writers[1] or writers[0] & writers[2] or writers[1] & writers[2])


def test_dataset_item(tmp_path):
    p = tmp_path / "w.png"
    Image.new("L", (120, 60), 255).save(p)
    cs = Charset()
    ds = IAMWords([WordItem("x", "x", "0", p, "Hi")], cs, Preprocess())
    x, target, n, text = ds[0]
    assert x.shape == (1, 32, 64) and n == 2 and text == "Hi"
    assert cs.decode(target.tolist()) == "Hi"


def test_aachen_split_files_assign_by_form():
    from pathlib import Path
    from src.config import DataConfig
    from src.data.iam import split_from_files
    cfg = DataConfig()
    sd = Path(cfg.split_dir)
    first = {s: (sd / f).read_text().split()[0] for s, f in cfg.split_files.items()}
    items = [WordItem(f"{form}-00-00", form, "w", None, "a") for form in first.values()]
    items.append(WordItem("zzz-999-00-00", "zzz-999", "w", None, "a"))   # in no list
    splits, missing = split_from_files(items, sd, cfg.split_files)
    for s, form in first.items():
        assert [i.form_id for i in splits[s]] == [form]
    assert missing == 1

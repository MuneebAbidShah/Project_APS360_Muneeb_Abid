"""The bundled form -> writer table must match IAM's published totals."""
from src.config import DataConfig
from src.data.iam import load_writers_csv


def test_table_matches_iam_totals():
    w = load_writers_csv(DataConfig().form_writers_csv)
    assert len(w) == 1539                 # IAM 3.0: 1,539 forms
    assert len(set(w.values())) == 657    # ... by 657 writers
    assert w["a01-000u"] == "000" and w["a01-000x"] == "001" and w["a01-003"] == "002"


def test_aachen_splits_are_writer_disjoint():
    cfg = DataConfig()
    w = load_writers_csv(cfg.form_writers_csv)
    ws = {}
    for s, f in cfg.split_files.items():
        forms = (cfg.split_dir / f).read_text().split()
        ws[s] = {w[x] for x in forms}
    assert (len(ws["train"]), len(ws["val"]), len(ws["test"])) == (283, 56, 161)
    assert not (ws["train"] & ws["val"] or ws["train"] & ws["test"] or ws["val"] & ws["test"])

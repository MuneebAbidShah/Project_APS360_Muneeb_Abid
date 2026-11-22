import pytest

from src.utils import metrics
from src.utils.metrics import _edit_distance_py, cer, edit_distance, wer


def test_edit_distance_known():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert _edit_distance_py("kitten", "sitting") == 3   # pure-Python fallback agrees


def test_cer_is_corpus_level():
    # 1 edit in "cat" + 2 edits in "house" = 3 edits over 3 + 5 = 8 characters
    assert cer(["cat", "house"], ["cut", "hose!"]) == pytest.approx(3 / 8)
    assert cer(["abc"], ["abc"]) == 0.0


def test_wer():
    assert wer(["cat", "dog", "bird"], ["cat", "dig", "bird"]) == pytest.approx(1 / 3)
    assert wer(["the cat sat"], ["the bat sat down"]) == pytest.approx(2 / 3)  # 1 sub + 1 insert


def test_fallback_without_editdistance(monkeypatch):
    monkeypatch.setattr(metrics, "_ed", None)
    assert metrics.cer(["cat"], ["cut"]) == pytest.approx(1 / 3)

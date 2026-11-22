import torch

from src.data.charset import BLANK, Charset


def test_blank_is_zero_and_class_count():
    cs = Charset("abc")
    assert BLANK == 0
    assert cs.encode("a") == [1]          # first real char is index 1
    assert cs.num_classes == 4            # 3 chars + blank


def test_round_trip_default_charset():
    cs = Charset()
    for word in ["Hello", "don't", "1960", "e.g.", '"Yes?"', "(it)"]:
        assert cs.decode(cs.encode(word)) == word


def test_is_valid():
    cs = Charset()
    assert cs.is_valid("abc123")
    assert not cs.is_valid("café")


def test_ctc_greedy_collapse():
    cs = Charset("abcdefghijklmnopqrstuvwxyz")
    c, a, t, l = (cs.encode(x)[0] for x in "catl")
    assert cs.ctc_greedy_decode([0, c, c, 0, a, a, t, 0]) == "cat"   # merge repeats, drop blanks
    assert cs.ctc_greedy_decode([l, 0, l]) == "ll"                   # blank separates double letters
    assert cs.ctc_greedy_decode([l, l, l]) == "l"
    assert cs.ctc_greedy_decode([0, 0, 0]) == ""


def test_decode_batch_from_logits():
    cs = Charset("ab")
    # (T=4, B=1, C=3) one-hot paths: a a blank b -> "ab"
    path = [1, 1, 0, 2]
    logits = torch.nn.functional.one_hot(torch.tensor(path), 3).float().unsqueeze(1)
    assert cs.decode_batch(logits) == ["ab"]

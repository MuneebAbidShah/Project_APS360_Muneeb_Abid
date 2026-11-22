"""PAGE mode (beta): segmenter, tracker and transcript. Duplicates must be removed by position, never by text."""
import os

import cv2
import numpy as np
import pytest

from webcam_helpers import word_images
from src.demo.page_reader import PageReader, PageSegmenter, Transcript, WordTracker, iou
from src.demo.synth import fake_page, page_on_desk


def jitter(b, rng, px=3):
    d = rng.integers(-px, px + 1, 4)
    return tuple(int(v) for v in np.array(b) + d)


HELLO, WORLD = (100, 100, 260, 150), (300, 102, 470, 152)


def test_same_word_over_30_frames_commits_once():
    t = WordTracker(commit_after=3)
    commits = sum(len(t.update([HELLO], ["hello"])) for _ in range(30))
    assert commits == 1 and len(t.committed) == 1 and len(t.tracks) == 1
    assert Transcript.text(t.committed) == "hello"


def test_pausing_adds_nothing():
    t = WordTracker()
    for _ in range(5):
        t.update([HELLO], ["hello"])
    before = [(x.id, x.text) for x in t.tracks]
    for _ in range(20):                         # writer pauses: same page, same readings (or flickering ones)
        t.update([HELLO], ["hello" if _ % 3 else "hallo"])
    for _ in range(10):                         # hand covers the page: nothing detected
        t.update([], [])
    assert [(x.id, x.text) for x in t.tracks] == before


def test_new_word_is_appended_in_reading_order():
    t = WordTracker()
    for _ in range(4):
        t.update([HELLO], ["hello"])
    for _ in range(4):
        t.update([HELLO, WORLD], ["hello", "world"])
    line2 = (100, 200, 200, 250)
    for _ in range(4):
        t.update([line2, HELLO, WORLD], ["again", "hello", "world"])
    assert Transcript.text(t.committed) == "hello world\nagain"


def test_jitter_and_shifts_create_no_duplicates():
    t, rng = WordTracker(), np.random.default_rng(0)
    for i in range(40):
        s = (i % 5) * 2                         # slow drift of the paper + per-frame jitter
        boxes = [jitter((b[0] + s, b[1] + s, b[2] + s, b[3] + s), rng) for b in (HELLO, WORLD)]
        t.update(boxes, ["word", "word"])       # identical TEXT for both: de-duplication is by position
    assert len(t.committed) == 2 and len(t.tracks) == 2
    assert Transcript.text(t.committed) == "word word"


def test_growing_word_updates_instead_of_appending():
    t = WordTracker()
    x0, y0, y1 = 100, 100, 150
    for x1, txt in [(150, "he"), (190, "hel"), (230, "hell"), (260, "hello")]:
        for _ in range(4):
            t.update([(x0, y0, x1, y1)], [txt])
    assert len(t.committed) == 1 and len(t.tracks) == 1
    assert t.committed[0].text == "hello" and t.committed[0].box[2] >= 255


def test_merged_box_over_two_committed_words_is_ignored():
    t = WordTracker()
    for _ in range(4):
        t.update([HELLO, WORLD], ["hello", "world"])
    for _ in range(6):
        t.update([(100, 100, 470, 152)], ["helloworld"])
    assert Transcript.text(t.committed) == "hello world" and len(t.tracks) == 2


def _page(seed=0):
    ims = word_images(seed=seed)
    lines = [ims[0:5], ims[5:10], ims[10:15]]
    return fake_page(lines, np.random.default_rng(seed), width=1500)


def _recall(found, truth):
    return np.mean([any(iou(tb, b) > 0.5 for b in found) for tb in truth])


def test_segmenter_on_a_synthetic_page_of_word_images():
    page, truth = _page()
    r = PageSegmenter().segment(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    flat = [b for row in truth for b in row]
    assert not r.paper_found                    # page fills the picture: no outline to straighten
    assert _recall(r.boxes, flat) >= 0.8 and abs(len(r.boxes) - len(flat)) <= 2
    assert len(set(r.lines)) == len(truth)
    for li, row in enumerate(truth):            # every true word lands on its own line
        for tb in row:
            hits = [l for b, l in zip(r.boxes, r.lines) if iou(tb, b) > 0.5]
            assert not hits or hits[0] == li


def test_segmenter_finds_and_straightens_the_paper():
    page, truth = _page(1)
    frame, _ = page_on_desk(page, np.random.default_rng(0))
    r = PageSegmenter().segment(frame)
    assert r.paper_found and r.page.shape[1] == 1000
    assert len(set(r.lines)) == len(truth)
    assert abs(len(r.boxes) - sum(len(t) for t in truth)) <= 3


class StubRecognizer:
    """Always reads 'word': the transcript can only be right if de-duplication is by position."""
    def tensor(self, g):
        import torch
        return torch.zeros(1, 32, max(16, g.shape[1] // 2))
    def predict_tensor(self, x):
        return "word"


def test_page_reader_live_writing_with_pauses_and_jitter():
    ims = word_images(seed=2)[:8]
    lines = [ims[:4], ims[4:8]]
    full, truth = fake_page(lines, np.random.default_rng(0), width=1500)
    order = [b for row in truth for b in row]
    reader, rng = PageReader(StubRecognizer()), np.random.default_rng(3)
    counts = []
    for k in range(1, len(order) + 1):          # words appear one at a time
        page = np.full_like(full, 245)
        for b in order[:k]:
            page[b[1]:b[3], b[0]:b[2]] = full[b[1]:b[3], b[0]:b[2]]
        for _ in range(6):                      # the writer pauses for 6 readings
            M = np.float32([[1, 0, rng.integers(-4, 5)], [0, 1, rng.integers(-4, 5)]])
            f = cv2.warpAffine(page, M, page.shape[::-1], borderValue=245)
            reader.process(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        counts.append(len(reader.tracker.committed))
    assert counts == sorted(counts)             # never goes down
    assert counts[-1] == len(order)             # every word exactly once
    assert reader.transcript() == "word word word word\nword word word word"


def test_page_reader_capture_rereads_every_crop():
    page, truth = _page()
    reader = PageReader(StubRecognizer())
    f = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    for _ in range(6):
        reader.process(f)
    assert any(c[2] is None for c in reader.last_crops)      # stable committed words aren't re-read every time
    reader.process(f, force_read=True)
    assert all(c[2] == "word" for c in reader.last_crops) and len(reader.last_crops) >= 12


IAM = os.environ.get("HTR_IAM_ROOT")


@pytest.mark.skipif(not IAM or not os.path.isdir(os.path.join(IAM or "", "words")), reason="needs real IAM images")
def test_segmenter_on_real_iam_word_images():
    import glob
    fs = sorted(glob.glob(os.path.join(IAM, "words", "*", "*", "*.png")))[:200]
    ims = [g for g in (cv2.imread(f, 0) for f in fs) if g is not None and g.size > 400][:30]
    lines = [ims[i:i + 5] for i in range(0, 30, 5)]
    page, truth = fake_page(lines, np.random.default_rng(0))
    r = PageSegmenter().segment(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    flat = [b for row in truth for b in row]
    assert _recall(r.boxes, flat) >= 0.75 and len(set(r.lines)) == len(truth)

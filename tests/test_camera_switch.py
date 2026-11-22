"""Camera switching (the 'press n and the demo closes' bug), the DroidCam placeholder detector,
--list and the crash handler, all with fake cameras (no real webcam needed)."""
import sys

import numpy as np
import pytest

from webcam_helpers import ROOT, tiny_checkpoint

sys.path.insert(0, str(ROOT))
from src.demo import live_demo as ld  # noqa: E402
from src.demo.webcam import export_model  # noqa: E402


class FakeCam:
    """Live = new sensor noise every frame; placeholder = the same picture every frame."""

    def __init__(self, bus, index, kind="live", shape=(480, 640)):
        self.bus, self.index, self.kind, self.released = bus, index, kind, False
        self.rng = np.random.default_rng(index)
        base = np.full(shape + (3,), 235, np.uint8)
        base[200:260, 150:500] = 40                           # a dark 'word'
        self.base = base
        if kind == "placeholder":                             # logo-like fixed screen
            self.base = np.full(shape + (3,), 30, np.uint8)
            self.base[180:300, 220:420] = (200, 120, 40)
        elif kind == "orange":                                # DroidCam Source 3 style flat screen
            self.base = np.full(shape + (3,), (0, 128, 255), np.uint8)

    def read(self):
        assert not self.released, "read() after release()"
        if self.kind == "live":
            noise = self.rng.normal(0, 3, self.base.shape)
            return True, np.clip(self.base + noise, 0, 255).astype(np.uint8)
        return True, self.base.copy()

    def release(self):
        self.released = True
        self.bus.open.discard(self.index)


class FakeBus:
    """open_camera() replacement. Like DirectShow, an index that is still open can't be opened
    again; `kinds` maps index -> 'live' / 'placeholder' / 'orange' (others don't exist)."""

    def __init__(self, kinds):
        self.kinds, self.open, self.calls = dict(kinds), set(), []

    def __call__(self, index, width=1280, height=720):
        self.calls.append(index)
        if index not in self.kinds or index in self.open:
            return None
        self.open.add(index)
        return FakeCam(self, index, self.kinds[index])


@pytest.fixture
def bus(monkeypatch):
    def make(kinds):
        b = FakeBus(kinds)
        monkeypatch.setattr(ld, "open_camera", b)
        monkeypatch.setattr(ld.time, "sleep", lambda s: None)
        return b
    return make


def test_switch_when_no_other_camera_reopens_the_previous_one(bus):
    b = bus({1: "placeholder"})
    src = ld.open_camera(1)
    new, idx, msg = ld.switch_camera(src, 1)
    assert new is not None and idx == 1 and ld.NO_OTHER_CAMERA in msg
    assert src.released and new.read()[0]                 # the old handle was released BEFORE probing
    assert set(b.calls[1:-1]) == {0, 2, 3, 4, 5}           # every other index 0-5 was tried


def test_switch_skips_failing_indices_and_wraps_round(bus):
    b = bus({0: "live", 4: "live"})
    src = ld.open_camera(4)
    new, idx, msg = ld.switch_camera(src, 4)
    assert idx == 0 and msg == "switched to camera 0"      # 5 fails, wraps to 0
    assert b.calls[1:] == [5, 0]
    new, idx, _ = ld.switch_camera(new, 0)
    assert idx == 4                                        # 1, 2, 3 fail, 4 works


def test_switch_never_returns_none_even_if_every_camera_fails(bus):
    b = bus({2: "live"})
    src = ld.open_camera(2)
    b.kinds.clear()                                        # e.g. DroidCam Client closed meanwhile
    new, idx, msg = ld.switch_camera(src, 2)
    assert isinstance(new, ld.BlankSource) and idx == 2 and "press n" in msg
    ok, frame = new.read()
    assert ok and frame.shape == (480, 640, 3)
    b.kinds[3] = "live"                                    # press n again once it's back
    new, idx, _ = ld.switch_camera(new, idx)
    assert idx == 3 and isinstance(new, FakeCam)


def test_probe_only_finds_free_cameras_so_it_must_run_with_none_open(bus):
    b = bus({0: "live", 1: "placeholder"})
    assert [c[0] for c in ld.probe_cameras()] == [0, 1]
    held = ld.open_camera(1)                               # the old bug: probing while camera 1 is open
    assert [c[0] for c in ld.probe_cameras()] == [0]
    held.release()


def test_still_detector_flags_placeholders_not_live_cameras(bus):
    bus({0: "live", 1: "placeholder", 2: "orange"})
    for idx, expect in ((0, False), (1, True), (2, True)):
        cam, det = ld.open_camera(idx), ld.StillDetector()
        flags = [det.add(cam.read()[1], t=k / 15) for k in range(20)]        # 20 frames at 15 fps
        assert flags[-1] is expect, idx
        cam.release()
    det = ld.StillDetector()                               # identical frames, but < 1 s: not yet
    f = np.full((480, 640, 3), 100, np.uint8)
    assert not any(det.add(f, t=k / 30) for k in range(10))
    det = ld.StillDetector()                               # blank WHITE paper (no colour) + noise: live
    rng = np.random.default_rng(0)
    for k in range(20):
        det.add(np.clip(240 + rng.normal(0, 2, (480, 640, 3)), 0, 255).astype(np.uint8), t=k / 15)
    assert not det.looks_placeholder


def test_list_reports_resolution_and_placeholders(bus, capsys):
    bus({0: "live", 1: "placeholder"})
    t = [0.0]

    def fake_time():
        t[0] += 0.07
        return t[0]
    ld.time.time, real = fake_time, ld.time.time
    try:
        cams = ld.main(["--list"])
    finally:
        ld.time.time = real
    out = capsys.readouterr().out
    assert [(c[0], c[1], c[2]) for c in cams] == [(0, 640, 480), (1, 640, 480)]
    assert "camera 0: 640x480, live picture" in out and "camera 1: 640x480, static" in out


@pytest.fixture
def model(tmp_path):
    ck = tmp_path / "best.pt"; tiny_checkpoint(ck)
    export_model(ck, tmp_path / "crnn_model.pt")
    return str(tmp_path / "crnn_model.pt")


def test_pressing_n_in_the_demo_never_crashes(bus, model, tmp_path, capsys):
    b = bus({1: "placeholder"})                            # the user's case: only DroidCam, not started
    ld.main(["--camera", "1", "--headless", "--model", model, "--out", str(tmp_path / "w"),
             "--keys", "..n..n..m.n..", "--max-frames", "14"])
    out = capsys.readouterr().out
    assert out.count(ld.NO_OTHER_CAMERA) == 3 and "trying camera 5" in out
    assert not b.open                                      # everything released at the end
    b = bus({0: "live", 1: "placeholder"})                 # a second camera exists: n switches to it
    ld.main(["--camera", "1", "--headless", "--model", model, "--out", str(tmp_path / "w"),
             "--keys", "..n...", "--max-frames", "6"])
    assert "switched to camera 0" in capsys.readouterr().out


def test_camera_that_stops_is_reopened_or_blanked(bus, model, tmp_path, capsys):
    b = bus({0: "live"})
    orig = FakeCam.read
    count = [0]

    def flaky(self):
        count[0] += 1
        if count[0] == 3:
            return False, None
        return orig(self)
    FakeCam.read = flaky
    try:
        ld.main(["--camera", "0", "--headless", "--model", model, "--out", str(tmp_path / "w"), "--max-frames", "6"])
    finally:
        FakeCam.read = orig
    assert "re-opening" in capsys.readouterr().out and not b.open


def test_crash_prints_traceback_instead_of_closing(monkeypatch, capsys):
    def boom(argv=None):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(ld, "main", boom)
    with pytest.raises(SystemExit) as e:
        ld.run(["--no-pause"])
    out = capsys.readouterr()
    assert e.value.code == 1 and "crashed" in out.out and "ValueError: synthetic failure" in out.err


def test_placeholder_hint_is_drawn():
    img = np.full((480, 640, 3), 200, np.uint8)
    ld.draw_hint(img, ld.PLACEHOLDER_HINT)
    assert img[62, :, 2].mean() > img[62, :, 0].mean() + 30    # red band under the header


# ------------------------------------------------------------------ camera names / --device-name

def test_preferred_order_by_device_name_then_droidcam():
    names = {0: "Integrated Webcam", 1: "OBS Virtual Camera", 2: "DroidCam Video", 3: "Some Phone Cam"}
    assert ld.preferred_order(range(6), names)[:2] == [2, 0]                     # DroidCam first by default
    assert ld.preferred_order(range(6), names, "phone")[:2] == [3, 2]            # then --device-name wins
    assert ld.preferred_order(range(6), names, "PHONE CAM")[0] == 3              # case-insensitive
    assert ld.preferred_order(range(6), {}, "x") == list(range(6))               # no names: number order


def test_startup_uses_the_matching_name_and_list_prints_names(bus, model, tmp_path, capsys, monkeypatch):
    bus({0: "live", 2: "live"})
    monkeypatch.setattr(ld, "camera_names", lambda: (["Integrated Webcam", "", "DroidCam Video"], True))
    ld.main(["--headless", "--model", model, "--out", str(tmp_path / "w"), "--max-frames", "2"])
    out = capsys.readouterr().out
    assert 'using 2 ("DroidCam Video")' in out
    ld.main(["--headless", "--model", model, "--out", str(tmp_path / "w"), "--max-frames", "2",
             "--device-name", "integrated"])
    assert 'using 0 ("Integrated Webcam")' in capsys.readouterr().out
    monkeypatch.setattr(ld.time, "time", iter(np.arange(0, 100, 0.07)).__next__)
    ld.main(["--list"])
    out = capsys.readouterr().out
    assert 'camera 0 "Integrated Webcam": 640x480, live picture' in out and 'camera 2 "DroidCam Video"' in out


def test_list_prints_unnumbered_windows_names(bus, capsys, monkeypatch):
    bus({1: "placeholder"})
    monkeypatch.setattr(ld, "camera_names", lambda: (["DroidCam Source 3", "USB Camera"], False))
    monkeypatch.setattr(ld.time, "time", iter(np.arange(0, 100, 0.07)).__next__)
    ld.main(["--list"])
    out = capsys.readouterr().out
    assert "- DroidCam Source 3" in out and "pygrabber" in out and "camera 1: 640x480, static" in out


def test_camera_names_never_raises(monkeypatch):
    monkeypatch.setattr(ld.sys, "platform", "win32")             # no pygrabber / powershell on this box
    names, indexed = ld.camera_names()
    assert isinstance(names, list) and isinstance(indexed, bool)

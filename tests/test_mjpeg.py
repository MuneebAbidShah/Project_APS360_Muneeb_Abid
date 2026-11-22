"""--droidcam: our own MJPEG-over-HTTP reader, tested against a local fake DroidCam server."""
import functools
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from webcam_helpers import ROOT, tiny_checkpoint

sys.path.insert(0, str(ROOT))
from src.demo import live_demo as ld  # noqa: E402
from src.demo.webcam import export_model  # noqa: E402


def scene(w, h, value=200):
    """A camera-like picture of paper: uneven lighting, a little noise, ruled lines, a word."""
    rng = np.random.default_rng(w)
    light = np.linspace(0.8, 1.1, w)[None, :, None] * np.linspace(0.9, 1.05, h)[:, None, None]
    img = np.clip(np.full((h, w, 3), value, np.float32) * light + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)
    for y in range(40, h, 32):                                   # ruled paper lines must NOT look scrambled
        cv2.line(img, (0, y), (w, y), (200, 170, 150), 1)
    cv2.putText(img, "could", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, w / 400, (20, 20, 20), max(2, w // 250))
    return img


@functools.lru_cache(maxsize=None)
def jpeg(w, h, value=200):
    return cv2.imencode(".jpg", scene(w, h, value))[1].tobytes()


def scrambled(w, h):
    """What a phone sends when its camera buffer is packed with the wrong width: a YUV buffer of a
    bigger picture read as w x h -> copies side by side + horizontal green/black lines."""
    big = cv2.cvtColor(scene(w * 3 // 2, h * 3 // 2), cv2.COLOR_BGR2YUV_I420).ravel()
    buf = np.resize(big, w * h * 3 // 2).reshape(h * 3 // 2, w)
    return cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_NV21)


@functools.lru_cache(maxsize=None)
def scrambled_jpeg(w, h):
    return cv2.imencode(".jpg", scrambled(w, h))[1].tobytes()


def size_for(path):
    return (640, 480) if "640x480" in path else (1280, 720) if "1280x720" in path else (1920, 1080)


class FakeDroidCam:
    """mode: 'ok' (multipart MJPEG), 'busy' (DroidCam's HTML 'busy' page), 'raw' (multipart parts that
    are raw camera bytes, not JPEG), 'rawnoparts' (a multipart content type, then only raw bytes).
    Size from the path: ?640x480, ?1280x720, else 1920x1080. frames_per_conn: close the connection
    after N frames (reconnect test). corrupt: a broken JPEG between good ones. only_paths: other
    paths 404. scrambled_paths: those paths send scrambled pictures. no_length: parts without
    Content-Length. shots: /cam/1/frame.jpg serves one JPEG. Stream paths not starting with
    /video or /mjpegfeed 404."""

    def __init__(self, mode="ok", frames_per_conn=None, corrupt=False, only_paths=None, scrambled_paths=(),
                 no_length=False, shots=False, single_viewer=False, busy_for=0.0):
        self.mode, self.frames_per_conn, self.corrupt, self.only_paths = mode, frames_per_conn, corrupt, only_paths
        self.scrambled_paths, self.no_length, self.shots = set(scrambled_paths), no_length, shots
        self.requests, self.connections, self.single_viewer, self.open_streams = [], 0, single_viewer, 0
        self.busy_until = time.time() + busy_for
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def nf(self):
                self.send_response(404); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(b"<html>Not found</html>")

            def do_GET(self):
                fake.requests.append(self.path)
                if self.path.startswith(("/cam/1/frame.jpg", "/shot.jpg")):
                    if not fake.shots:
                        return self.nf()
                    body = jpeg(1280, 720, 210)
                    self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body))); self.end_headers()
                    self.wfile.write(body); return
                if fake.only_paths is not None and self.path not in fake.only_paths:
                    return self.nf()
                if not self.path.startswith(("/video", "/mjpegfeed")):
                    return self.nf()
                if fake.mode == "busy" or time.time() < fake.busy_until or (fake.single_viewer and fake.open_streams > 0):
                    self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                    self.wfile.write(b"<html><body>DroidCam is Busy</body></html>"); return
                fake.connections += 1
                fake.open_streams += 1
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--dcmjpeg")
                self.end_headers()
                w, h = size_for(self.path)
                k = 0
                try:
                    while fake.frames_per_conn is None or k < fake.frames_per_conn:
                        if fake.mode == "rawnoparts":
                            self.wfile.write(np.random.randint(0, 255, w * 3 * 64, np.uint8).tobytes())
                            self.wfile.flush(); k += 1; time.sleep(0.005); continue
                        if fake.mode == "raw":                          # a raw NV21 buffer, not a JPEG
                            body = np.random.randint(0, 255, w * h * 3 // 2, np.uint8).tobytes()
                        elif fake.corrupt and k % 2 == 1:
                            body = b"\xff\xd8" + b"\x00garbage" * 200 + b"\xff\xd9"
                        elif self.path in fake.scrambled_paths:
                            body = scrambled_jpeg(w, h)
                        else:
                            body = jpeg(w, h, 200 + k % 3)
                        head = b"--dcmjpeg\r\nContent-Type: image/jpeg\r\n"
                        if not fake.no_length:
                            head += b"Content-Length: %d\r\n" % len(body)
                        self.wfile.write(head + b"\r\n" + body + b"\r\n")
                        self.wfile.flush(); k += 1
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    if fake.single_viewer:               # like the phone: lets go a bit after the close
                        time.sleep(1.0)
                    fake.open_streams -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


@pytest.fixture
def droid():
    servers = []

    def make(**kw):
        s = FakeDroidCam(**kw); servers.append(s); return s
    yield make
    for s in servers:
        s.close()


def test_urls_prefer_the_phone_default_size_and_accept_ip_port_or_url():
    assert ld.droidcam_urls("10.0.0.74") == ["http://10.0.0.74:4747" + p for p in
                                             ("/video", "/video?1280x720", "/mjpegfeed", "/video?640x480")]
    assert ld.droidcam_urls("10.0.0.74:8080")[0] == "http://10.0.0.74:8080/video"
    assert ld.droidcam_urls("http://10.0.0.74:4747/video") == ["http://10.0.0.74:4747/video"]
    assert ld.droidcam_urls("http://10.0.0.74:4747")[0].endswith(":4747/video")
    assert ld.droidcam_shot_urls("10.0.0.74") == ["http://10.0.0.74:4747/cam/1/frame.jpg",
                                                  "http://10.0.0.74:4747/shot.jpg"]


def test_scrambled_detector():
    for w, h in ((1920, 1080), (1280, 720), (640, 480)):
        assert not ld.looks_garbled(cv2.imdecode(np.frombuffer(jpeg(w, h), np.uint8), 1))
        assert ld.looks_garbled(cv2.imdecode(np.frombuffer(scrambled_jpeg(w, h), np.uint8), 1))
    noisy = np.clip(scene(1280, 720).astype(np.int16) + np.random.normal(0, 8, (720, 1280, 3)), 0, 255)
    assert not ld.looks_garbled(noisy.astype(np.uint8))


def test_decode_jpeg_only_accepts_complete_jpegs():
    good = jpeg(640, 480)
    assert ld.decode_jpeg(good).shape == (480, 640, 3)
    assert ld.decode_jpeg(good + b"\r\n").shape == (480, 640, 3)
    assert ld.decode_jpeg(good[: len(good) // 2]) is None                 # cut off
    raw = np.random.randint(0, 255, 640 * 480 * 3 // 2, np.uint8).tobytes()
    assert ld.decode_jpeg(raw) is None                                     # raw camera buffer
    assert ld.decode_jpeg(b"\xff\xd8" + raw + b"\xff\xd9") is None       # JPEG markers around junk


def test_decodes_frames_from_the_default_path_and_prints_the_check_line(droid, capsys):
    s = droid()
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, verbose=False)
    try:
        ok, f = src.read()
        assert ok and f.shape == (1080, 1920, 3)
        assert s.requests[0] == "/video" and src.url.endswith("/video")
        time.sleep(0.4)
        assert src.count > 3 and 5 < src.fps < 80                          # real picture rate (~40/s here)
        out = capsys.readouterr().out
        assert "DroidCam check:" in out and "type=multipart/x-mixed-replace" in out
        assert "first bytes=ff d8" in out and "1920x1080 JPEG (row/col change" in out and ") ok" in out
    finally:
        src.release()


def test_fps_counts_new_pictures_not_reads(droid):
    s = droid(frames_per_conn=8)
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}/video", start_wait=5, verbose=False)
    try:
        s.frames_per_conn = 0                                             # the phone stops sending
        time.sleep(src.stale + 0.5)
        for _ in range(50):
            src.read()
        assert src.fps == 0.0
    finally:
        src.release()


def test_falls_back_to_other_paths_and_never_asks_a_404_again(droid):
    s = droid(only_paths={"/video?640x480"})
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, verbose=False, gap=0.1)
    try:
        assert src.url.endswith("/video?640x480") and src.read()[1].shape == (480, 640, 3)
        assert s.requests[:4] == list(ld.DROIDCAM_PATHS)
        s.frames_per_conn = 3; time.sleep(1.0)                           # force a few reconnects
        assert all(r == "/video?640x480" for r in s.requests[4:])
    finally:
        src.release()


def test_scrambled_stream_stays_open_and_is_shown_without_reconnecting(droid, capsys, tmp_path):
    """The real phone: /video gives scrambled-looking JPEGs and is the only viewer it allows."""
    s = droid(scrambled_paths={"/video"}, single_viewer=True, only_paths={"/video", "/video?1280x720"})
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=8, sample_path=str(tmp_path / "sample.jpg"))
    try:
        assert src.url.endswith("/video") and src.read()[1].shape == (1080, 1920, 3)
        c0 = src.count
        time.sleep(1.0)
        assert src.count > c0 + 10                                        # still streaming
        assert s.requests == ["/video"] and s.connections == 1           # never hopped / reconnected
        assert src.garbled and src.status == ld.GARBLED_NOTE
        out = capsys.readouterr().out
        assert "look SCRAMBLED -> showing it anyway" in out and "sample.jpg" in out and "BUSY" not in out
    finally:
        src.release()


def test_waits_for_a_busy_phone_to_free_up(droid, capsys):
    s = droid(busy_for=3.0, only_paths={"/video"})
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=10, gap=1.0)
    try:
        assert src.read()[1].shape == (1080, 1920, 3)
        assert 2 <= s.requests.count("/video") <= 4                        # a pause between tries, no hammering
        assert "still trying every few seconds" in capsys.readouterr().out
    finally:
        src.release()


def test_no_garble_check_flag(droid, capsys):
    s = droid(scrambled_paths={"/video"})
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, garble_check=False)
    try:
        time.sleep(0.3)
        assert not src.garbled and src.status == ""
        assert "ok (scramble check off)" in capsys.readouterr().out
    finally:
        src.release()


def test_first_picture_is_saved_exactly_as_sent(droid, tmp_path, capsys):
    s = droid()
    p = tmp_path / "live" / "droidcam_sample.jpg"
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, sample_path=str(p))
    try:
        time.sleep(0.3)
        data = p.read_bytes()
        assert data in {jpeg(1920, 1080, 200 + k) for k in range(3)}      # byte-for-byte the phone's JPEG
        assert str(p) in capsys.readouterr().out and src.sample_saved == str(p)
        assert cv2.imread(str(p)).shape == (1080, 1920, 3)
    finally:
        src.release()


def test_raw_non_jpeg_bytes_are_rejected(droid, capsys):
    s = droid(mode="raw")
    with pytest.raises(SystemExit) as e:
        ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=8, gap=0.1)
    assert "not with JPEG pictures" in str(e.value.code)
    out = capsys.readouterr().out
    assert "REJECTED" in out and "none a complete JPEG" in out


def test_raw_bytes_without_parts_are_rejected(droid, capsys):
    s = droid(mode="rawnoparts", only_paths={"/video"})
    with pytest.raises(SystemExit):
        ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=8, gap=0.1)
    assert "without any picture part" in capsys.readouterr().out


def test_raw_stream_falls_back_to_single_pictures(droid, capsys):
    s = droid(mode="raw", shots=True)
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=10, gap=0.1)
    try:
        assert src.url.endswith("/cam/1/frame.jpg") and src.read()[1].shape == (720, 1280, 3)
        time.sleep(0.5)
        assert src.count > 3
        assert "using single pictures" in capsys.readouterr().out
    finally:
        src.release()


def test_parts_without_content_length(droid):
    s = droid(no_length=True)
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, verbose=False)
    try:
        time.sleep(0.3)
        assert src.read()[1].shape == (1080, 1920, 3) and src.count > 3 and src.bad == 0
    finally:
        src.release()


def test_corrupted_frames_are_dropped(droid):
    s = droid(corrupt=True)
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, verbose=False)
    try:
        time.sleep(0.4)
        assert src.bad >= 2 and src.count >= 2 and src.read()[1].shape == (1080, 1920, 3)
    finally:
        src.release()


def test_reconnects_when_the_stream_drops(droid, capsys):
    s = droid(frames_per_conn=10)
    src = ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=5, gap=0.1)
    try:
        t0 = time.time()
        while s.connections < 3 and time.time() - t0 < 10:
            src.read(); time.sleep(0.05)
        assert s.connections >= 3 and src.count >= 12        # kept getting frames after reconnecting
    finally:
        src.release()


def test_busy_reply_prints_the_fix(droid, capsys):
    s = droid(mode="busy")
    with pytest.raises(SystemExit) as e:
        ld.MJPEGSource(f"127.0.0.1:{s.port}", start_wait=6)
    msg = str(e.value.code)
    assert "BUSY" in msg and "Quit the DroidCam Client" in msg and "browser tab" in msg
    assert "DroidCam Client" in capsys.readouterr().out


def test_unreachable_phone_gives_same_wifi_advice():
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    with pytest.raises(SystemExit) as e:
        ld.MJPEGSource(f"127.0.0.1:{port}", start_wait=1.5, timeout=0.5, verbose=False)
    assert "SAME Wi-Fi" in str(e.value.code)


def test_demo_runs_on_droidcam_and_on_http_image(droid, tmp_path, capsys):
    ck = tmp_path / "best.pt"; tiny_checkpoint(ck); export_model(ck, tmp_path / "m.pt")
    s = droid()
    common = ["--headless", "--model", str(tmp_path / "m.pt"), "--out", str(tmp_path / "w"), "--max-frames", "4"]
    ld.main(common + ["--droidcam", f"127.0.0.1:{s.port}", "--keys", ".n..",       # n must not crash
                      "--save-frame", str(tmp_path / "droidcam_sample.jpg")])
    out = capsys.readouterr().out
    assert "DroidCam stream" in out and "1920x1080" in out and "FINAL" in out and "DroidCam check:" in out
    assert (tmp_path / "droidcam_sample.jpg").exists()
    ld.main(common + ["--image", f"http://127.0.0.1:{s.port}/video?640x480"])
    assert "640x480" in capsys.readouterr().out


def test_backend_choice():
    assert ld.backends("msmf") == [cv2.CAP_MSMF] and ld.backends("dshow") == [cv2.CAP_DSHOW]
    assert ld.backends("any") == [cv2.CAP_ANY]
    ld_platform = ld.sys.platform
    try:
        ld.sys.platform = "win32"
        assert ld.backends("auto") == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    finally:
        ld.sys.platform = ld_platform

"""Live handwriting demo: point a webcam (or a phone via DroidCam) at a word written on paper.

    python live_demo.py                 # finds a camera (tries 0-5)
    python live_demo.py --camera 1      # pick one (DroidCam is often 0 or 1)
    python live_demo.py --list          # list the cameras (name, live or placeholder), then quit
    python live_demo.py --device-name droidcam   # prefer the camera whose name contains this text
    python live_demo.py --droidcam 10.0.0.74     # read the phone app DIRECTLY over Wi-Fi (no Client)

Two modes, switched with m (the current mode is shown at the top left):

WORD mode (default, the reliable one): one word in the green guide box.
    q / Esc      quit
    space        freeze / unfreeze the picture (the frozen word is read once)
    c            capture: freezes, then TYPE THE TRUE WORD and press Enter to save it to
                 webcam_test/ (Esc cancels). Used to build the ~150-word webcam test set.
    a / d        guide box narrower / wider       w / s   guide box taller / shorter
    n            switch to the next camera that works
    m            switch to PAGE mode

PAGE mode (BETA, less accurate): a running transcript of a whole page (written live, or an
old notebook page). About twice a second the page is found, straightened, cut into lines and
words, and every word is read; a word is added to the transcript (side panel) only once its box
has been stable for a few readings (green = in the transcript, yellow = not yet).
    r            reset the transcript         s   save the transcript to transcripts/*.txt
    space        freeze / unfreeze            c   page capture: save EVERY word crop of the
                 (frozen) page into webcam_test/ with the model's guess as the label and
                 needs_check=1. Fix those labels in labels.csv and set needs_check to 0:
                 rows still marked 1 are NOT scored (a guess is not a true label).
    m            back to WORD mode            q   quit

The word inside the green box is cleaned up exactly like in the Step 5 notebook
(src/demo/webcam.py: flatten the lighting, find the ink, tight crop, IAM-like contrast), then
resized and normalized with the SAME transform as training (src/data/transforms.py), and read by
the CRNN on the CPU every few frames. The small window shows exactly what the model sees.
Testing without a camera:  python live_demo.py --image frame.png --headless
"""
import argparse
import collections
import csv
import os
import sys
import time
import traceback

# OpenCV prints pages of warnings for every camera index that doesn't exist; this demo prints its
# own clear messages instead (set OPENCV_LOG_LEVEL=WARNING to see OpenCV's again).
os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
HERE = os.path.dirname(os.path.abspath(__file__))
for _root in (HERE, os.path.abspath(os.path.join(HERE, "..", ".."))):   # live_demo folder, or the repo
    if os.path.isdir(os.path.join(_root, "src", "demo")):
        sys.path.insert(0, _root)
        break

import numpy as np  # noqa: E402

KEY_ESC, KEY_ENTER, KEY_ENTER2, KEY_BACK, KEY_BACK2, KEY_SPACE = 27, 13, 10, 8, 127, 32
LABELS_CSV = "labels.csv"
FIELDS = ["file", "label", "prediction", "camera", "saved_at", "frame_w", "frame_h", "needs_check"]
TARGET_WORDS = 150


# ------------------------------------------------------------------ small pure-Python pieces (tested)

class Smoother:
    """Most common prediction over the last k readings (ties -> the most recent), so the
    overlay doesn't flicker between two spellings."""

    def __init__(self, k=5):
        self.hist = collections.deque(maxlen=k)

    def add(self, text):
        self.hist.append(text)
        return self.value()

    def value(self):
        if not self.hist:
            return ""
        counts = collections.Counter(self.hist)
        best = max(counts.values())
        for t in reversed(self.hist):
            if counts[t] == best:
                return t

    def clear(self):
        self.hist.clear()


class Keys:
    """What each key does. view = 'word' or 'page'; mode = 'live', 'frozen' or 'label' (typing the
    true word, word view only). handle(key) returns an action name; typed text is in self.text."""

    WORD_KEYS = {"a": "narrower", "d": "wider", "w": "taller", "s": "shorter", "n": "next_camera", "c": "label"}
    PAGE_KEYS = {"r": "reset", "s": "save_text", "c": "page_capture", "n": "next_camera"}

    def __init__(self, view="word"):
        self.view, self.mode, self.text = view, "live", ""

    def handle(self, key):
        if key < 0 or key == 255:
            return None
        if self.mode == "label":
            if key == KEY_ESC:
                self.mode, self.text = "frozen", ""
                return "cancel"
            if key in (KEY_ENTER, KEY_ENTER2):
                if not self.text.strip():
                    return "empty"
                return "save"
            if key in (KEY_BACK, KEY_BACK2):
                self.text = self.text[:-1]
                return "edit"
            if 33 <= key <= 126:                 # printable ASCII (IAM words have no spaces)
                self.text += chr(key)
                return "edit"
            return None
        ch = chr(key).lower() if key < 256 else ""
        if key == KEY_ESC or ch == "q":
            return "quit"
        if key == KEY_SPACE:
            self.mode = "frozen" if self.mode == "live" else "live"
            return "freeze" if self.mode == "frozen" else "unfreeze"
        if ch == "m":
            self.view, self.mode = ("page" if self.view == "word" else "word"), "live"
            return "mode"
        act = (self.WORD_KEYS if self.view == "word" else self.PAGE_KEYS).get(ch)
        if act == "label":
            self.mode, self.text = "label", ""
        elif act == "page_capture":
            self.mode = "frozen"
        return act

    def saved(self):
        self.mode, self.text = "live", ""


def count_saved(out_dir):
    p = os.path.join(out_dir, LABELS_CSV)
    if not os.path.exists(p):
        return 0
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _ensure_header(path):
    """An older labels.csv without the newest columns is rewritten with the current header."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames == FIELDS:
            return
        rows = list(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({**{k: "" for k in FIELDS}, **row, "needs_check": row.get("needs_check") or "0"})


def save_capture(out_dir, crop_bgr, label, prediction="", camera="", frame_shape=(0, 0), needs_check=False):
    """Save the raw guide-box crop as images/NNNN.png and append a row to labels.csv.
    The raw crop is kept (not the cleaned one) so the notebook can redo the clean-up.
    needs_check=True marks a label that is only the model's guess (page capture): not scored until fixed.
    Returns (file name relative to out_dir, number of words saved so far)."""
    import cv2
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(img_dir, f"{n:04d}.png")):
        n += 1
    rel = f"images/{n:04d}.png"
    if not cv2.imwrite(os.path.join(out_dir, rel), crop_bgr):
        raise IOError(f"could not write {rel}")
    path = os.path.join(out_dir, LABELS_CSV)
    _ensure_header(path)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"file": rel, "label": label.strip(), "prediction": prediction, "camera": camera,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_w": frame_shape[1], "frame_h": frame_shape[0], "needs_check": int(bool(needs_check))})
    return rel, count_saved(out_dir)


# ------------------------------------------------------------------ cameras and fake sources

BACKEND = "auto"          # set from --backend


def backends(choice=None):
    """OpenCV capture APIs to try, in order. auto = DirectShow, then Media Foundation, then any
    (on Windows); just 'any' elsewhere."""
    import cv2
    choice = (choice or BACKEND).lower()
    named = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    if choice in named:
        return [named[choice]]
    return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]


def backend():
    return backends()[0]


def open_camera(index, width=1280, height=720):
    """An opened camera that really gives frames, or None. Tries the --backend APIs in order."""
    import cv2
    for api in backends():
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):                  # some cameras (DroidCam) need a few frames to start
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return cap
            time.sleep(0.05)
        cap.release()
    return None


MAX_CAMERAS = 6        # indices 0-5 are probed (DroidCam is often 1 or 2 when a laptop webcam exists)
NO_OTHER_CAMERA = "no other camera gives a picture"
PLACEHOLDER_HINT = ("This looks like the DroidCam placeholder: open the DroidCam Client, press Start,",
                    "make sure it shows your phone video")


DEFAULT_PREFER = "droidcam"     # DroidCam's Windows camera is called e.g. "DroidCam Video" / "DroidCam Source 3"


def camera_names():
    """(names, indexed). indexed=True: names[i] is the name of camera index i (the DirectShow order
    OpenCV uses on Windows, read with the optional 'pygrabber' package; /sys on Linux).
    indexed=False: just the camera names Windows knows (PowerShell), without index numbers.
    ([], False) if nothing could be read. Never raises."""
    try:
        if sys.platform.startswith("win"):
            try:
                from pygrabber.dshow_graph import FilterGraph
                return list(FilterGraph().get_input_devices()), True
            except Exception:
                pass
            import subprocess
            ps = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or "
                  "$_.PNPClass -eq 'Image' } | ForEach-Object { $_.Name }")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                               timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return [x.strip() for x in r.stdout.splitlines() if x.strip()], False
        names, root = [], "/sys/class/video4linux"
        for i in range(MAX_CAMERAS):
            p = os.path.join(root, f"video{i}", "name")
            names.append(open(p).read().strip() if os.path.exists(p) else "")
        return (names, True) if any(names) else ([], False)
    except Exception:
        return [], False


def name_matches(name, wanted):
    return bool(name) and bool(wanted) and wanted.lower() in name.lower()


def preferred_order(indices, names, device_name=None):
    """Camera indices to try, best first: a name containing --device-name, then one containing
    'DroidCam', then the rest in number order. Case-insensitive; names = {index: name}."""
    idx = list(indices)

    def rank(i):
        n = names.get(i, "")
        return (0 if name_matches(n, device_name) else 1 if name_matches(n, DEFAULT_PREFER) else 2, i)
    return sorted(idx, key=rank)


def indexed_names(names, indexed):
    return {i: n for i, n in enumerate(names) if n} if indexed else {}


def probe_cameras(indices=range(MAX_CAMERAS)):
    """[(index, width, height)] for every camera index that gives frames. Each camera is released
    before the next is tried; call this only while no camera is open (DSHOW can't open a busy one)."""
    found = []
    for i in indices:
        cap = _open_or_none(i)
        if cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((i, frame.shape[1], frame.shape[0]))
            cap.release()
    return found


class StillDetector:
    """Spots the DroidCam placeholder. A real camera never gives the same picture twice (sensor
    noise), while DroidCam's 'not connected' output is a fixed image (its logo screen, or a flat
    light-blue / orange / green screen). looks_placeholder is True when the frames of the last
    ~1 s are all (nearly) identical, or a frame is one saturated flat colour (white paper is
    not saturated, so a blank sheet doesn't trigger it)."""

    def __init__(self, seconds=1.0, min_frames=5, thr=0.1, flat_frac=0.9):
        self.seconds, self.min_frames, self.thr, self.flat_frac = seconds, min_frames, thr, flat_frac
        self.reset()

    def reset(self):
        self.hist = collections.deque()
        self.static = self.flat = False

    @staticmethod
    def flat_colour(frame, frac=0.9):
        """True if >= frac of the pixels are one saturated colour (DroidCam's coloured screens)."""
        import cv2
        if frame.ndim != 3:
            return False
        hsv = cv2.cvtColor(np.ascontiguousarray(frame[::4, ::4]), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
        med = np.median(hsv, axis=0)
        if med[1] < 60:                                    # grey / white / black: not a coloured screen
            return False
        dh = np.abs(hsv[:, 0] - med[0]); dh = np.minimum(dh, 180 - dh)
        close = (dh <= 8) & (np.abs(hsv[:, 1] - med[1]) <= 40) & (np.abs(hsv[:, 2] - med[2]) <= 40)
        return bool(close.mean() >= frac)

    def add(self, frame, t=None):
        t = time.time() if t is None else t
        g = frame if frame.ndim == 2 else frame.mean(axis=2)
        small = np.asarray(g[::4, ::4], np.float32)           # subsample, no averaging: keeps the noise
        if self.hist and self.hist[-1][1].shape != small.shape:
            self.hist.clear()
        self.hist.append((t, small))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.seconds:
            self.hist.popleft()                                # keep just over `seconds` of frames
        span = t - self.hist[0][0]
        self.static = (len(self.hist) >= self.min_frames and span >= self.seconds and
                       max(float(np.abs(f - small).mean()) for _, f in self.hist) < self.thr)
        self.flat = self.flat_colour(frame, self.flat_frac)
        return self.looks_placeholder

    @property
    def looks_placeholder(self):
        return self.static or self.flat


def describe_cameras(indices=range(MAX_CAMERAS), seconds=1.2):
    """For --list: [(index, width, height, verdict)], verdict = live picture or placeholder-like."""
    out = []
    for i in indices:
        cap = _open_or_none(i)
        if cap is None:
            continue
        det, shape, t0 = StillDetector(), None, time.time()
        try:
            while time.time() - t0 < seconds:
                ok, frame = cap.read()
                if ok and frame is not None:
                    shape = frame.shape
                    det.add(frame)
                time.sleep(0.03)
        finally:
            cap.release()
        if shape is None:
            continue
        verdict = ("flat colour: looks like the DroidCam placeholder" if det.flat else
                   "static picture: looks like the DroidCam placeholder (Client not started?)" if det.static else
                   "live picture")
        out.append((i, shape[1], shape[0], verdict))
    return out


class BlankSource:
    """Stand-in when no camera opens at all, so the program keeps running (press n to retry)."""

    def __init__(self, text="no camera gives a picture: press n to retry, q to quit", shape=(480, 640)):
        self.img = np.full(shape + (3,), 40, np.uint8)
        put_text(self.img, text, (10, shape[0] // 2), 0.55)

    def read(self):
        return True, self.img.copy()

    def release(self):
        pass


def _open_or_none(i):
    try:
        return open_camera(i)
    except Exception as e:                       # a broken driver must not kill the demo
        print(f"  camera {i}: error while opening ({e})")
        return None


def switch_camera(src, current, indices=range(MAX_CAMERAS), retries=3, pause=0.2, names=None):
    """Release `src`, then try the indices after `current` in turn (wrapping round). Returns
    (new_source, new_index, message). Never returns None as the source: if no other index gives
    frames the previous camera is re-opened (message NO_OTHER_CAMERA); if even that fails, a
    BlankSource that says 'press n to retry'."""
    if src is not None:
        try:
            src.release()
        except Exception:
            pass
        time.sleep(pause)                        # DSHOW needs a moment before the device is free again
    idx = list(indices)
    start = idx.index(current) + 1 if current in idx else 0
    order = [i for i in idx[start:] + idx[:start] if i != current]
    for i in order:
        print(f"  trying camera {i} ...")
        cap = _open_or_none(i)
        if cap is not None:
            label = f' "{names[i]}"' if names and i in names else ""
            print(f"switched to camera {i}{label}")
            return cap, i, f"switched to camera {i}{label}"
    if current is not None:
        for _ in range(retries):
            cap = _open_or_none(current)
            if cap is not None:
                print(f"{NO_OTHER_CAMERA}: staying on camera {current}")
                return cap, current, f"{NO_OTHER_CAMERA} (still camera {current})"
            time.sleep(pause)
    print("no camera gives a picture at all: check the DroidCam Client, then press n to retry")
    return BlankSource(), current, "no camera gives a picture: press n to retry"


# ------------------------------------------------------------------ DroidCam over Wi-Fi (MJPEG over HTTP)

DROIDCAM_PORT = 4747
# The phone's own default size first (a requested size can make some phones send a scrambled
# picture). Addresses that answer 404 are dropped after one try, so the extra ones cost nothing.
DROIDCAM_PATHS = ("/video", "/video?1280x720", "/mjpegfeed", "/video?640x480")
DROIDCAM_SHOTS = ("/cam/1/frame.jpg", "/shot.jpg")       # one JPEG per request (slower fallback)
CHECK_FRAMES = 6            # frames looked at on connect to decide if a URL's picture is scrambled
BUSY_HELP = ("The phone's DroidCam app says it is BUSY (another program is using the video). Fix:\n"
             "  1) Quit the DroidCam Client on the laptop completely: right-click its icon in the system tray\n"
             "     (bottom right, near the clock, maybe under the ^ arrow) -> Exit; or Task Manager -> DroidCam -> End task.\n"
             "  2) Close every browser tab that shows the phone video.\n"
             "  3) On the phone, close the DroidCam app and open it again.\n"
             "  Then run this again.")
NET_HELP = ("Can't reach the phone at {url}. Check:\n"
            "  - the phone and the laptop are on the SAME Wi-Fi (not a guest network, no VPN),\n"
            "  - the DroidCam app is open on the phone with the screen on,\n"
            "  - the IP is exactly the 'WiFi IP' shown in the app (it can change after a reconnect).")
NOJPEG_HELP = ("The phone answered, but not with JPEG pictures this program can read.\n"
               "  - In the DroidCam phone app: Settings -> set the video format/quality to the defaults\n"
               "    (Quality 'Normal', Advanced Options OFF), close the app and open it again.\n"
               "  - Please copy the 'DroidCam check' lines above when asking for help.")
GARBLED_HELP = ("The phone's picture looks SCRAMBLED (copies side by side, green lines). It is shown anyway.\n"
                "  - If the demo window actually looks FINE, ignore this (or run with --no-garble-check).\n"
                "  - Please send the saved picture {sample} when asking for help.\n"
                "  - If it really is scrambled: close and reopen the DroidCam app, switch to the back camera,\n"
                "    set its video Quality to 'Normal', or restart the phone.")
GARBLED_NOTE = "phone picture looks scrambled? see the terminal"


def is_stream_url(s):
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def droidcam_urls(target, paths=DROIDCAM_PATHS):
    """'10.0.0.74', '10.0.0.74:4747' or a full URL -> URLs to try, best first. A URL with its own
    path is used as given; without a path the DroidCam paths (phone default size first) are tried."""
    t = target.strip()
    if not is_stream_url(t):
        t = "http://" + t
    from urllib.parse import urlsplit
    u = urlsplit(t)
    host = u.hostname or ""
    port = u.port or DROIDCAM_PORT
    base = f"{u.scheme}://{host}:{port}"
    if u.path not in ("", "/") or u.query:
        return [base + u.path + (("?" + u.query) if u.query else "")]
    return [base + p for p in paths]


def droidcam_shot_urls(target):
    """Single-picture URLs on the same phone (used when the video stream can't be read)."""
    base = droidcam_urls(target)[0].split("/", 3)
    return ["/".join(base[:3]) + p for p in DROIDCAM_SHOTS]


def decode_jpeg(data):
    """Bytes -> BGR image, ONLY if they are one complete JPEG (starts FFD8, ends FFD9) that OpenCV
    decodes. Anything else (raw camera buffers, cut-off or broken JPEGs) gives None: raw bytes are
    never reshaped into a picture."""
    import cv2
    data = bytes(data).rstrip(b"\r\n \t")
    if len(data) < 200 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or min(img.shape[:2]) < 32:
        return None
    return img


def garble_scores(img):
    """(typical change between neighbouring rows, ... between neighbouring columns), in grey levels."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, max(2, g.shape[0] * 640 // g.shape[1])), interpolation=cv2.INTER_NEAREST)
    g = g.astype(np.int16)
    rows = float(np.median(np.abs(np.diff(g, axis=0)).mean(1)))
    cols = float(np.median(np.abs(np.diff(g, axis=1)).mean(0)))
    return rows, cols


def looks_garbled(img):
    """True for the 'scrambled' picture a phone sends when its camera buffer is packed with the wrong
    width: the scene repeated side by side and covered in horizontal lines. Natural pictures change
    about as much from one row to the next as from one column to the next; those change far more
    between rows. Medians are used, so ruled paper lines or text don't count."""
    rows, cols = garble_scores(img)
    return rows > 2.0 and (rows + 1) / (cols + 1) > 2.0


class _BadStream(Exception):
    pass


class MJPEGSource:
    """Reads the DroidCam phone app over Wi-Fi with urllib on a background thread that keeps only
    the newest frame, so the demo never lags behind.
    - Multipart MJPEG: each part is cut out by its Content-Length (or, without one, by the next
      boundary) and must be one complete JPEG (FFD8 ... FFD9) that cv2.imdecode accepts; anything
      else is dropped. Raw bytes are never reshaped into a picture.
    - The FIRST stream that gives real JPEGs is kept open and shown, even if its picture looks
      scrambled (then a note is shown and advice printed): the phone serves one viewer at a time,
      so hopping between URLs only makes it 'busy'. Addresses that answer 404 are never asked
      again; between real connections there is a pause of `gap` seconds.
    - Only when no stream works are single JPEG pictures polled (/cam/1/frame.jpg, /shot.jpg).
    - The first picture is saved exactly as the phone sent it (sample_path), to look at / send.
    - One 'DroidCam check' line per URL tried: URL, Content-Type, first bytes, result.
    - fps counts NEW decoded frames only. Reconnects by itself. read() returns the newest frame."""

    def __init__(self, target, timeout=5.0, start_wait=20.0, stale=3.0, verbose=True,
                 garble_check=True, sample_path=None, gap=2.5, busy_wait=12.0):
        import threading
        self.urls, self.timeout, self.stale, self.verbose = droidcam_urls(target), timeout, stale, verbose
        self.shots = droidcam_shot_urls(target) if len(self.urls) > 1 else []
        self.garble_check, self.sample_path, self.gap = garble_check, sample_path, gap
        self.url, self.good_url, self.dead = self.urls[0], None, set()
        self.frame, self.t_frame, self.count, self.bad, self.connections = None, 0.0, 0, 0, 0
        self.garbled, self.sample_saved, self._raw_first, self._t_conn = False, None, b"", 0.0
        self.times = collections.deque(maxlen=30)
        self.error, self.status, self._said, self.checks = "", "", set(), []
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        t0 = time.time()
        while time.time() - t0 < start_wait and self.frame is None and self.thread.is_alive():
            if self.error == "busy" and time.time() - t0 > min(busy_wait, start_wait):
                break
            time.sleep(0.05)
        if self.frame is None:
            self.release()
            raise SystemExit(self.advice() + "\n(no picture from the phone)")
        h, w = self.frame.shape[:2]
        print(f"DroidCam stream: {self.url} ({w}x{h})")

    @property
    def fps(self):
        with self.lock:
            t = list(self.times)
        if len(t) < 2 or time.time() - t[-1] > self.stale:
            return 0.0
        return (len(t) - 1) / max(1e-6, t[-1] - t[0])

    def advice(self):
        if self.error == "busy":
            return BUSY_HELP
        if self.error == "html":
            return "The phone answered with a web page instead of video.\n" + BUSY_HELP
        if self.error == "nojpeg":
            return NOJPEG_HELP
        if self.error == "notfound":
            return "The phone answered, but has no video at these addresses (HTTP 404).\n" + NOJPEG_HELP
        return NET_HELP.format(url=self.urls[0].split("/", 3)[2])

    def _say(self, key, msg):
        if self.verbose and key not in self._said:
            print(msg)
            self._said.add(key)

    def _check(self, url, ctype, first, result):
        """The one diagnostic line per URL (always printed: it is what we ask the user to paste)."""
        line = (f"DroidCam check: {url} | type={ctype or '-'} | first bytes={first[:12].hex(' ') or '-'} "
                f"| {result}")
        self.checks.append(line)
        if ("check", url, result) not in self._said:
            self._said.add(("check", url, result))
            print(line)

    def _store(self, img, url):
        with self.lock:
            self.frame, self.t_frame = img, time.time()
            self.times.append(self.t_frame)
            self.count += 1
        self.url, self.error = url, ""
        self.status = GARBLED_NOTE if self.garbled else ""
        self._said.discard("lost")

    def _save_sample(self, data):
        if not self.sample_path or self.sample_saved:
            return
        try:
            d = os.path.dirname(os.path.abspath(self.sample_path))
            os.makedirs(d, exist_ok=True)
            with open(self.sample_path, "wb") as f:
                f.write(bytes(data).rstrip(b"\r\n \t"))
            self.sample_saved = os.path.abspath(self.sample_path)
            print(f"saved the first phone picture (exactly as the phone sent it) to: {self.sample_saved}")
        except OSError as e:
            print(f"(couldn't save the sample picture to {self.sample_path}: {e})")

    def _wait_gap(self):
        """Pause between real connections so the phone has time to let go of the last one."""
        left = self._t_conn + self.gap - time.time()
        if left > 0:
            self.stop.wait(left)

    def _run(self):
        shots_tried = False
        while not self.stop.is_set():
            urls = [self.good_url] if self.good_url else [u for u in self.urls if u not in self.dead]
            for url in urls:
                if self.stop.is_set():
                    return
                if self._attempt(url) == "ok":
                    self.good_url = url                   # had pictures: always come back to this one
                    break
            else:
                if self.frame is None and not self.good_url and self.shots and not shots_tried:
                    shots_tried = True
                    if self._poll_shots():
                        return
                if not urls and not self.good_url:        # everything 404: keep trying the first slowly
                    self.dead.discard(self.urls[0])
            if self.stop.is_set():
                return
            if self.frame is not None:
                self.status = self.status or "phone stream lost: reconnecting ..."
            self.stop.wait(0.2)
            self._wait_gap()

    def _attempt(self, url):
        import socket
        import urllib.error
        self._wait_gap()
        try:
            return self._open(url)
        except urllib.error.HTTPError as e:
            try:
                body = e.read(4000)
            except Exception:
                body = b""
            ctype = e.headers.get("Content-Type", "") if e.headers else ""
            self._check(url, ctype, body, f"HTTP {e.code}" + (" (not on this phone, not asked again)" if e.code == 404 else ""))
            if e.code == 404:
                self.dead.add(url)
                if self.frame is None and self.error in ("", "network"):
                    self.error = "notfound"
                return "dead"
            self._t_conn = time.time()
            self._bad_reply(url, body, e.code)
            return "bad"
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            self._t_conn = time.time()
            if self.frame is None:
                self.error = "network"
            self._say(("net", str(getattr(e, "reason", e))),
                      f"can't connect to {url} ({getattr(e, 'reason', e)})\n" + NET_HELP.format(url=url.split("/", 3)[2]))
            return "net"

    def _open(self, url):
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "aps360-live-demo"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self._t_conn = time.time()
            ctype = resp.headers.get("Content-Type") or ""
            low = ctype.lower()
            if "multipart" not in low and not low.startswith("image/"):
                body = resp.read(4000)
                self._check(url, ctype, body, "not a video stream")
                self._bad_reply(url, body)
                return "bad"
            self.connections += 1
            if "multipart" not in low:               # a single picture
                body = resp.read()
                img = decode_jpeg(body)
                if img is None:
                    self.error = self.error or "nojpeg"
                    self._check(url, ctype, body, "REJECTED: not a complete JPEG")
                    return "nojpeg"
                self._save_sample(body)
                self._judge(img, url, ctype, body)
                self._store(img, url)
                return "ok"
            return self._pump(resp, url, ctype)

    def _judge(self, img, url, ctype, first, n=1, k=None):
        sz = f"{img.shape[1]}x{img.shape[0]} JPEG"
        if not self.garble_check:
            self._check(url, ctype, first, f"{sz} ok (scramble check off)")
            return
        k = int(looks_garbled(img)) if k is None else k
        self.garbled = k * 2 > n
        r, c = garble_scores(img)
        sz += f" (row/col change {r:.1f}/{c:.1f})"
        if self.garbled:
            self._check(url, ctype, first, f"{sz}, {k}/{n} frames look SCRAMBLED -> showing it anyway")
            self._say("garbledhelp", GARBLED_HELP.format(sample=self.sample_saved or "droidcam_sample.jpg"))
        else:
            self._check(url, ctype, first, f"{sz} ok")

    def _parts(self, resp, ctype, limit=3_000_000):
        """Yield the payload bytes of each multipart part (cut by Content-Length, or by the next
        boundary when there is none). Bytes that aren't inside a part are thrown away; too many of
        them means this isn't a multipart picture stream at all."""
        import re
        m = re.search(r'boundary="?([^";]+)"?', ctype, re.I)
        token = m.group(1).strip().lstrip("-").encode() if m else b""
        buf, junk = b"", 0
        while not self.stop.is_set():
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(16384)
            if not chunk:
                return
            if not buf and not junk and not self._raw_first:
                self._raw_first = chunk[:16]
            buf += chunk
            while True:
                h = buf.find(b"\r\n\r\n")
                if h < 0 or h > 2000:                    # no part header in sight
                    if h > 2000 or len(buf) > 4000:
                        cut = h + 4 if h > 0 else len(buf) - 3
                        junk += cut
                        buf = buf[cut:]
                        if junk > limit:
                            raise _BadStream(f"{junk // 1000} kB without any picture part")
                        continue
                    break
                head = buf[:h].decode("latin-1", "replace").lower()
                if "content-type" not in head and "content-length" not in head:
                    junk += h + 4
                    buf = buf[h + 4:]
                    if junk > limit:
                        raise _BadStream(f"{junk // 1000} kB without any picture part")
                    continue
                n = re.search(r"content-length:\s*(\d+)", head)
                start = h + 4
                if n:
                    end = start + int(n.group(1))
                    if len(buf) < end:
                        break
                    data, buf = buf[start:end], buf[end:]
                else:
                    nxt = buf.find(b"--" + token, start) if token else buf.find(b"\r\n--", start)
                    if nxt < 0:
                        if len(buf) - start > limit:
                            raise _BadStream("a part with no end")
                        break
                    data, buf = buf[start:nxt], buf[nxt:]
                yield data

    def _pump(self, resp, url, ctype):
        """Show every good JPEG of this stream until it ends. Never closes it because of how the
        picture looks: the first CHECK_FRAMES pictures only decide the note and the check line."""
        seen, good, first, k, last = 0, 0, b"", 0, None
        self._raw_first = b""
        judged = url == self.good_url and ("judged", url) in self._said
        try:
            for data in self._parts(resp, ctype):
                if not first:
                    first = data[:16] or b"?"
                seen += 1
                img = decode_jpeg(data)
                if img is None:
                    self.bad += 1
                    if good == 0 and seen >= 3 * CHECK_FRAMES:
                        self.error = self.error or "nojpeg"
                        self._check(url, ctype, first, f"REJECTED: {seen} parts, none a complete JPEG")
                        return "nojpeg"
                    continue
                good, last = good + 1, img
                if good == 1:
                    self._save_sample(data)
                if not judged:
                    if self.garble_check:
                        k += looks_garbled(img)
                    if good >= (CHECK_FRAMES if self.garble_check else 1):
                        self._judge(img, url, ctype, first, good, k)
                        judged = True
                        self._said.add(("judged", url))
                self._store(img, url)
        except _BadStream as e:
            self._check(url, ctype, first or self._raw_first, f"REJECTED: {e}")
            self.error = self.error or "nojpeg"
            return "nojpeg"
        if good and not judged:                   # stream ended before CHECK_FRAMES pictures
            self._judge(last, url, ctype, first, good, k)
            self._said.add(("judged", url))
        if not good:
            self._check(url, ctype, first or self._raw_first, f"REJECTED: {seen} parts, none a complete JPEG")
            self.error = self.error or "nojpeg"
            return "nojpeg"
        return "ok"

    def _poll_shots(self):
        """Fallback when no stream works: ask for one JPEG at a time. True if it ran (until release())."""
        import urllib.error
        import urllib.request
        for url in self.shots:
            if url in self.dead:
                continue
            if self._attempt(url) != "ok":
                continue
            print(f"using single pictures from {url} (slower than video, but readable)")
            misses = 0
            while not self.stop.is_set():
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "aps360-live-demo"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        img = decode_jpeg(r.read())
                    if img is None:
                        self.bad += 1
                    else:
                        self._store(img, url)
                        misses = 0
                        continue
                except (urllib.error.URLError, OSError):
                    misses += 1
                    self.status = "phone stream lost: reconnecting ..."
                    self.stop.wait(min(self.gap, 0.3 * misses))
            return True
        return False

    def _bad_reply(self, url, body, code=None):
        text = body.decode("utf-8", "replace").lower()
        err = "busy" if "busy" in text else "html" if ("<" in text or code) else "other"
        if self.frame is None or err == "busy":
            self.error = err
        snippet = " ".join(text.split())[:100]
        self._say(("reply", url, err), f"{url}: not a video stream ({'HTTP ' + str(code) + ', ' if code else ''}"
                                       f"'{snippet}')")
        if err == "busy":
            self._say("busyhelp", BUSY_HELP + "\n  (still trying every few seconds ...)")

    def read(self):
        with self.lock:
            f, t = self.frame, self.t_frame
        if f is None:
            return False, None
        if time.time() - t > self.stale:
            self.status = self.status if self.status and self.status != GARBLED_NOTE else "phone stream lost: reconnecting ..."
            self._say("lost", "phone stream lost: reconnecting ... (is the DroidCam app still open?)")
        return True, f.copy()

    def release(self):
        self.stop.set()
        if self.thread.is_alive() and self.thread is not __import__("threading").current_thread():
            self.thread.join(timeout=1.0)


class FileSource:
    """A picture or video file used as the 'camera' (for testing without a webcam)."""

    def __init__(self, path):
        import cv2
        self.img = cv2.imread(path) if os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp") else None
        self.cap = None if self.img is not None else cv2.VideoCapture(path)
        if self.img is None and not self.cap.isOpened():
            raise SystemExit(f"Can't open {path}")

    def read(self):
        if self.img is not None:
            return True, self.img.copy()
        ok, f = self.cap.read()
        if not ok:                                   # loop the video
            self.cap.set(1, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ------------------------------------------------------------------ drawing

def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2, outline=True):
    import cv2
    if outline:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def model_view(model_u8, scale=3, min_w=240):
    """The 32-px model input, enlarged (nearest neighbour) for the side window."""
    import cv2
    if model_u8 is None:
        v = np.full((32 * scale, min_w), 128, np.uint8)
        put_text(v, "no writing in the box", (8, 32 * scale // 2 + 6), 0.5)
        return v
    v = cv2.resize(model_u8, (model_u8.shape[1] * scale, model_u8.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    if v.shape[1] < min_w:
        v = np.hstack([v, np.full((v.shape[0], min_w - v.shape[1]), 200, np.uint8)])
    return v


def _header(out, keys, fps, saved):
    mode = "WORD mode" if keys.view == "word" else "PAGE mode (BETA)"
    put_text(out, f"{mode}   {fps:4.1f} FPS" + ("   FROZEN" if keys.mode != "live" else ""), (10, 28), 0.7,
             (255, 255, 255) if keys.view == "word" else (0, 220, 255))
    put_text(out, f"webcam_test: {saved} / {TARGET_WORDS} saved   (m = switch mode)", (10, 56), 0.6, (200, 255, 200))


def draw(frame, box, keys, shown, fps, saved, found_ink, note=""):
    """WORD mode overlay."""
    import cv2
    out = frame.copy()
    x0, y0, x1, y1 = box
    color = (0, 200, 0) if found_ink else (0, 200, 255)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    put_text(out, shown if shown else "(nothing read)", (x0, max(80, y0 - 12)), 1.1, (255, 255, 0), 2)
    _header(out, keys, fps, saved)
    h = out.shape[0]
    if keys.mode == "label":
        put_text(out, "Type the TRUE word, Enter = save, Esc = cancel:", (10, h - 48), 0.6, (0, 255, 255))
        put_text(out, keys.text + "_", (10, h - 16), 0.9, (0, 255, 255))
    else:
        put_text(out, "q quit | space freeze | c capture+label | a/d w/s box size | n camera | m page mode",
                 (10, h - 16), 0.5)
    if note:
        put_text(out, note, (10, h - 76), 0.6, (0, 255, 255))
    return out


def draw_page(frame, reader, keys, fps, saved, note=""):
    """PAGE mode overlay: committed words green, pending yellow; transcript in a side panel."""
    import cv2
    out = frame.copy()
    for t in reader.tracker.tracks:
        if t.missed > 0 and not t.committed:
            continue
        pts = reader.frame_box(t.box)
        cv2.polylines(out, [pts], True, (0, 200, 0) if t.committed else (0, 220, 255), 2)
    _header(out, keys, fps, saved)
    h = out.shape[0]
    put_text(out, "q quit | space freeze | r reset | s save text | c page capture | m word mode", (10, h - 16), 0.5)
    if note:
        put_text(out, note, (10, h - 48), 0.6, (0, 255, 255))
    return np.hstack([out, transcript_panel(reader, h)])


def draw_hint(img, lines, scale=0.6, top=58):
    """Warning lines on a dark-red band just under the header (above the guide box, clear of the
    key help at the bottom)."""
    import cv2
    w = img.shape[1]
    lh = int(26 * scale / 0.6)
    band = img[top:top + lh * len(lines) + 8]
    band[:] = (band * 0.25 + np.array([0, 0, 140]) * 0.75).astype(np.uint8)
    for k, line in enumerate(lines):
        s = scale
        while s > 0.3 and cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][0] > w - 20:
            s -= 0.05
        put_text(img, line, (10, top + lh * (k + 1)), s, (255, 255, 255), 1, outline=False)
    return img


def transcript_panel(reader, h, w=420):
    import textwrap
    p = np.full((h, w, 3), 250, np.uint8)
    put_text(p, "Transcript (PAGE mode, BETA)", (10, 28), 0.6, (0, 120, 255), 1, outline=False)
    n_c, n_p = len(reader.tracker.committed), len(reader.tracker.pending)
    put_text(p, f"{n_c} words | {n_p} pending", (10, 54), 0.5, (90, 90, 90), 1, outline=False)
    lines = []
    for row in reader.transcript().split("\n"):
        lines += textwrap.wrap(row, 30) or [""]
    per = max(1, (h - 80) // 26)
    for i, line in enumerate(lines[-per:]):
        put_text(p, line, (10, 84 + 26 * i), 0.65, (30, 30, 30), 1, outline=False)
    return p


def save_transcript(reader, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, time.strftime("page_%Y%m%d_%H%M%S.txt"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(reader.transcript() + "\n")
    return path


def page_capture(reader, out_dir, cam_name, frame_shape):
    """Save every word crop of the last processed page with the model's guess as label (needs_check=1)."""
    n = 0
    for _, crop_bgr, guess in reader.last_crops:
        if crop_bgr.size and guess:
            save_capture(out_dir, crop_bgr, guess, guess, cam_name + " page", frame_shape, needs_check=True)
            n += 1
    return n, count_saved(out_dir)


# ------------------------------------------------------------------ main loop

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: first that works of 0-5)")
    ap.add_argument("--list", action="store_true", help="list cameras (name, resolution, live or placeholder) and quit")
    ap.add_argument("--device-name", default=None,
                    help="prefer the camera whose Windows name contains this text (any case), e.g. DroidCam "
                         "or part of your phone's name; names containing 'DroidCam' are preferred anyway")
    ap.add_argument("--model", default=os.path.join(HERE, "crnn_model.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "webcam_test"), help="where captures are saved")
    ap.add_argument("--transcripts", default=os.path.join(HERE, "transcripts"), help="where page transcripts go")
    ap.add_argument("--every", type=int, default=3, help="WORD mode: run the model every N frames")
    ap.add_argument("--page-interval", type=float, default=0.5, help="PAGE mode: seconds between page readings")
    ap.add_argument("--smooth", type=int, default=5, help="majority vote over the last N readings")
    ap.add_argument("--box", type=float, nargs=2, default=(0.7, 0.25), metavar=("W", "H"),
                    help="guide box size as a fraction of the frame")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270), help="rotate the camera picture")
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--page", action="store_true", help="start in PAGE mode (beta)")
    ap.add_argument("--droidcam", metavar="IP[:PORT]",
                    help="read the DroidCam phone app directly over Wi-Fi (no DroidCam Client needed), "
                         "e.g. --droidcam 10.0.0.74 (port 4747 by default) or a full http:// URL")
    ap.add_argument("--save-frame", default=os.path.join(HERE, "droidcam_sample.jpg"), metavar="PATH",
                    help="--droidcam: save the first phone picture, exactly as the phone sent it, here "
                         "(default: droidcam_sample.jpg next to live_demo.py; '' = don't save)")
    ap.add_argument("--no-garble-check", action="store_true",
                    help="--droidcam: don't check whether the phone's picture looks scrambled")
    ap.add_argument("--backend", default="auto", choices=("auto", "dshow", "msmf", "any"),
                    help="OpenCV camera API (auto = DirectShow, then Media Foundation, then any)")
    ap.add_argument("--image", help="use a picture or video file (or an http:// MJPEG stream) instead of a camera")
    ap.add_argument("--headless", action="store_true", help="no windows: print predictions (testing)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--capture-label", help="headless testing: save the last crop with this label")
    ap.add_argument("--keys", default="", help="headless testing: one key per frame ('.' = no key)")
    ap.add_argument("--no-pause", action="store_true", help="don't wait for Enter after an error")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import cv2
    from src.demo.page_reader import PageReader
    from src.demo.webcam import Recognizer, crop, guide_box
    global BACKEND
    BACKEND = a.backend
    names, indexed = camera_names()
    by_index = indexed_names(names, indexed)
    if a.list:
        print(f"checking cameras 0-{MAX_CAMERAS - 1} (about 1 s each) ...")
        cams = describe_cameras()
        for i, w, h, verdict in cams:
            label = f' "{by_index[i]}"' if i in by_index else ""
            print(f"  camera {i}{label}: {w}x{h}, {verdict}")
        if not cams:
            print("  no camera gives a picture (is the DroidCam Client started?)")
        if names and not indexed:
            print("camera names Windows reports (their numbers are unknown; `python -m pip install pygrabber` "
                  "shows which number is which):")
            for nm in names:
                print("   -", nm)
        elif not names:
            print("(camera names not available)")
        print("use one with:  python live_demo.py --camera N   or   python live_demo.py --device-name DroidCam")
        return cams
    if not os.path.exists(a.model):
        raise SystemExit(f"Model file not found: {a.model}\nKeep live_demo.py next to crnn_model.pt (unzip the whole folder).")
    rec = Recognizer(a.model, threads=a.threads)
    reader = PageReader(rec)
    print(f"model loaded ({sum(p.numel() for p in rec.model.parameters()):,} parameters, CPU)")

    cams = []
    stream = a.droidcam or (a.image if is_stream_url(a.image) else None)
    is_cam = not (a.image or a.droidcam)
    if stream:
        print(f"connecting to the phone: {droidcam_urls(stream)[0].split('/', 3)[2]} ...")
        src = MJPEGSource(stream, garble_check=not a.no_garble_check, sample_path=a.save_frame or None)
        cam_name = "droidcam " + droidcam_urls(stream)[0].split("/", 3)[2]
    elif a.image:
        src, cam_name = FileSource(a.image), os.path.basename(a.image)
    else:
        if a.camera is None:
            if by_index:
                print("cameras:", ", ".join(f'{i} "{n}"' for i, n in sorted(by_index.items())))
            if a.device_name and by_index and not any(name_matches(n, a.device_name) for n in by_index.values()):
                print(f"no camera name contains '{a.device_name}'; trying the others")
            order = preferred_order(range(MAX_CAMERAS), by_index, a.device_name)
            print(f"looking for cameras 0-{MAX_CAMERAS - 1} ...")
            cams = [c[0] for c in probe_cameras(order)]
            if not cams:
                raise SystemExit("No camera found. Is the DroidCam Client connected (it must say 'connected' "
                                 "and show the phone picture)? Then run again, or try --camera 0 / 1 / 2.")
            a.camera = cams[0]
            why = f' ("{by_index[a.camera]}")' if a.camera in by_index else ""
            print(f"working cameras: {sorted(cams)} -> using {a.camera}{why} "
                  "(press n in the window to switch, or use --camera N)")
        else:
            cams = [a.camera]
        src = open_camera(a.camera)
        if src is None:
            raise SystemExit(f"Camera {a.camera} doesn't give a picture. Try another: python live_demo.py --list")
        cam_name = f"camera {a.camera}"
        print(f"using camera {a.camera}" + (f' "{by_index[a.camera]}"' if a.camera in by_index else ""))

    keys, smooth = Keys("page" if a.page else "word"), Smoother(a.smooth)
    wfrac, hfrac = a.box
    saved = count_saved(a.out)
    frozen, last_crop, raw, shown, model_u8, found = None, None, "", "", None, False
    fps, t_prev, n, note, note_until, t_page = 0.0, time.time(), 0, "", 0, 0.0
    interval = 0.0 if a.headless else a.page_interval
    still, warned = StillDetector(), False
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(a.rotate)
    if not a.headless:
        cv2.namedWindow("Live demo", cv2.WINDOW_NORMAL)
        cv2.namedWindow("What the model sees", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("What the model sees", 20, 20)

    def read_word(frame, box):
        c = crop(frame, box)
        t, mu, g = rec.read(c)
        return c, t, mu, g is not None

    try:
        while True:
            if keys.mode == "live" or frozen is None:
                ok, frame = src.read()
                if (not ok or frame is None) and is_cam:
                    print(f"camera {a.camera} stopped giving frames: re-opening it ...")
                    try:
                        src.release()
                    except Exception:
                        pass
                    src = _open_or_none(a.camera) or BlankSource()
                    ok, frame = src.read()
                if not ok or frame is None:
                    print("the picture source stopped giving frames"); break
                if is_cam and still.add(frame) and not warned:
                    print("WARNING: " + " ".join(PLACEHOLDER_HINT) + " (or press n for the next camera)")
                    warned = True
                if rot is not None:
                    frame = cv2.rotate(frame, rot)
                frozen = frame if keys.mode != "live" else None
            else:
                frame = frozen
            box = guide_box(frame.shape, wfrac, hfrac)
            n += 1
            now = time.time()
            if keys.view == "word":
                if keys.mode == "live" and (n - 1) % max(1, a.every) == 0:
                    last_crop, raw, model_u8, found = read_word(frame, box)
                    shown = smooth.add(raw if found else "")
                    if a.headless:
                        print(f"frame {n}: '{shown}' (this frame '{raw}')")
                elif keys.mode != "live" and last_crop is None:
                    last_crop, raw, model_u8, found = read_word(frame, box); shown = raw
            elif keys.mode == "live" and now - t_page >= interval:
                reader.process(frame); t_page = now
                model_u8 = reader.last_model_input
                if a.headless:
                    print(f"frame {n}: {len(reader.tracker.committed)} words committed, "
                          f"{len(reader.tracker.pending)} pending")
            if stream:                                     # count NEW phone pictures, not loop turns
                fps = src.fps
            else:
                fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev) if fps else 1 / max(1e-6, now - t_prev)
            t_prev = now
            if a.headless:
                if a.max_frames and n >= a.max_frames:
                    break
                k = a.keys[n - 1] if n - 1 < len(a.keys) else "."
                act = None if k == "." else keys.handle(ord(k))
            else:
                msg = note if now < note_until else ""
                view = draw(frame, box, keys, shown, fps, saved, found, msg) if keys.view == "word" else \
                    draw_page(frame, reader, keys, fps, saved, msg)
                if still.looks_placeholder and is_cam:
                    draw_hint(view, PLACEHOLDER_HINT + ("(or press n to try the next camera)",))
                elif getattr(src, "status", ""):
                    draw_hint(view, (src.status,))
                cv2.imshow("Live demo", view)
                cv2.imshow("What the model sees", model_view(model_u8))
                act = keys.handle(cv2.waitKey(1))
                if cv2.getWindowProperty("Live demo", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if act == "quit":
                break
            if act in ("freeze", "label"):
                frozen = frame
                if keys.view == "word":                  # read the frozen word once, without smoothing
                    last_crop, raw, model_u8, found = read_word(frame, box); shown = raw
                else:
                    reader.process(frame)
            elif act == "unfreeze" or act == "mode":
                smooth.clear(); last_crop = None
                if act == "mode":
                    note, note_until = ("PAGE mode is BETA: less accurate than WORD mode" if keys.view == "page"
                                        else "WORD mode"), now + 3
            elif act == "save":
                rel, saved = save_capture(a.out, last_crop, keys.text, raw, cam_name, frame.shape)
                note, note_until = f"saved {rel} as '{keys.text.strip()}'", now + 2.5
                print(f"saved {rel}: '{keys.text.strip()}' (model said '{raw}') -> {saved} words")
                keys.saved(); smooth.clear()
            elif act == "empty":
                note, note_until = "type the word first (Esc cancels)", now + 2
            elif act == "reset":
                reader.reset(); note, note_until = "transcript cleared", now + 2
            elif act == "save_text":
                p = save_transcript(reader, a.transcripts)
                note, note_until = f"saved {os.path.basename(p)}", now + 3
                print("transcript saved to", p)
            elif act == "page_capture":
                frozen = frame
                reader.process(frame, force_read=True)
                k, saved = page_capture(reader, a.out, cam_name, frame.shape)
                note, note_until = f"saved {k} word crops with GUESSED labels: fix them in labels.csv", now + 5
                print(f"page capture: {k} crops saved with the model's guesses as labels (needs_check=1). "
                      f"Edit {os.path.join(a.out, LABELS_CSV)}: correct each label and set needs_check to 0.")
            elif act in ("narrower", "wider"):
                wfrac = float(np.clip(wfrac + (0.05 if act == "wider" else -0.05), 0.2, 0.95))
            elif act in ("taller", "shorter"):
                hfrac = float(np.clip(hfrac + (0.03 if act == "taller" else -0.03), 0.08, 0.6))
            elif act == "next_camera" and not is_cam:
                note, note_until = ("reading the phone over Wi-Fi: n does nothing here" if stream else
                                    "reading a file: n does nothing here"), time.time() + 2
            elif act == "next_camera":
                print(f"n pressed: looking for another camera than {a.camera} (0-{MAX_CAMERAS - 1}) ...")
                src, a.camera, msg = switch_camera(src, a.camera, names=by_index)
                cam_name = f"camera {a.camera}"; smooth.clear(); reader.reset(); still.reset()
                warned, frozen, last_crop = False, None, None
                keys.mode = "live"
                note, note_until = msg, time.time() + 3
    finally:
        if src is not None:
            try:
                src.release()
            except Exception:
                pass
        if not a.headless:
            cv2.destroyAllWindows()
    if keys.view == "page":
        print("TRANSCRIPT:\n" + reader.transcript())
        return reader.transcript()
    if a.headless and a.capture_label and last_crop is not None:
        rel, saved = save_capture(a.out, last_crop, a.capture_label, raw, cam_name, frame.shape)
        print(f"saved {rel} -> {saved} words")
    print(f"FINAL: '{shown}'")
    return shown


def run(argv=None):
    """main(), but an error never just closes the window: the full traceback is printed and the
    program waits for Enter, so it stays readable (e.g. in the VS Code terminal)."""
    no_pause = "--no-pause" in (sys.argv[1:] if argv is None else argv)

    def wait():
        if not no_pause and sys.stdin is not None and sys.stdin.isatty():
            try:
                input("\nPress Enter to close ...")
            except EOFError:
                pass
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    except SystemExit as e:
        if e.code not in (None, 0):
            if isinstance(e.code, str):
                print(e.code)
            wait()
            sys.exit(1)
        raise
    except Exception:
        print("\n*** live_demo.py crashed. Please copy everything below when asking for help: ***\n")
        traceback.print_exc()
        wait()
        sys.exit(1)


if __name__ == "__main__":
    run()

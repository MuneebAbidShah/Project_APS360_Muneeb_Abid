"""Make a fake webcam_test/ folder (for testing Step 5 without a camera): IAM word images are
turned into simulated phone photos (src/demo/synth.py) and saved exactly like live_demo.py's
'c' capture does (images/NNNN.png + labels.csv).

    python scripts/make_fake_webcam.py --iam-root data/iam --out /tmp/webcam_test --n 40
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.demo.live_demo import save_capture  # noqa: E402
from src.demo.synth import fake_frame  # noqa: E402
from src.demo.webcam import crop  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from src.data.iam import parse_words_txt, image_path
    words = [(w, t) for w, st, t in parse_words_txt(Path(a.iam_root) / "ascii" / "words.txt")
             if st == "ok" and image_path(Path(a.iam_root), w).exists() and t.isalpha() and len(t) >= 2]
    rng = np.random.default_rng(a.seed)
    for wid, text in random.Random(a.seed).sample(words, min(a.n, len(words))):
        g = cv2.imread(str(image_path(Path(a.iam_root), wid)), 0)
        if g is None:
            continue
        frame, box = fake_frame(g, rng)
        save_capture(a.out, crop(frame, box), text, "", "fake", frame.shape)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

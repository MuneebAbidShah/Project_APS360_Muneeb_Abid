"""Fake webcam frames for testing: an IAM word "written on paper and filmed by a phone".

Paper colour, uneven light, a larger word with lighter/blurred ink, slight rotation and
perspective, sensor noise, JPEG compression and (sometimes) ruled lines. Used by the tests,
the mock run and `scripts/make_fake_webcam.py`; real captures come from live_demo.py.
"""
import numpy as np

from src.demo.webcam import guide_box


def fake_frame(word_gray, rng=None, frame_hw=(480, 640), box=(0.7, 0.25), lines=None):
    """Return (BGR uint8 frame, guide box). The word is placed inside the guide box."""
    import cv2
    rng = rng or np.random.default_rng(0)
    H, W = frame_hw
    x0, y0, x1, y1 = guide_box((H, W), *box)
    # paper with a lighting gradient and a soft vignette
    paper = rng.uniform(165, 215)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 1 + rng.uniform(-0.18, 0.18) * (xx / W - 0.5) + rng.uniform(-0.18, 0.18) * (yy / H - 0.5)
    vign = 1 - 0.25 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    light = grad * vign
    canvas = np.full((H, W), paper, np.float32)
    if lines if lines is not None else rng.random() < 0.4:            # ruled paper
        gap = int(rng.uniform(0.09, 0.14) * H)
        for y in range(int(rng.uniform(0, gap)), H, gap):
            canvas[y:y + 2] -= rng.uniform(25, 45)
    # the word: scaled to ~45-80% of the box height, ink lighter and blurred
    g = word_gray.astype(np.float32)
    target_h = rng.uniform(0.45, 0.8) * (y1 - y0)
    s = min(target_h / g.shape[0], 0.9 * (x1 - x0) / g.shape[1])
    g = cv2.resize(g, (max(2, int(g.shape[1] * s)), max(2, int(g.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip((250 - g) / 190, 0, 1)                            # how much ink at each pixel
    ink = rng.uniform(40, 110)
    h, w = g.shape
    cy = int((y0 + y1) / 2 + rng.uniform(-0.1, 0.1) * (y1 - y0)) - h // 2
    cx = int((x0 + x1) / 2 + rng.uniform(-0.05, 0.05) * (x1 - x0)) - w // 2
    cy, cx = max(0, min(H - h, cy)), max(0, min(W - w, cx))
    region = canvas[cy:cy + h, cx:cx + w]
    canvas[cy:cy + h, cx:cx + w] = region * (1 - alpha) + ink * alpha
    canvas = canvas * light
    # camera: small rotation + perspective, blur, noise, colour cast, JPEG
    ang = rng.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    canvas = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    d = rng.uniform(-0.02, 0.02, (4, 2)) * [W, H]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, np.float32(src + d)), (W, H),
                                 borderMode=cv2.BORDER_REPLICATE)
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.6, 1.4))
    canvas += rng.normal(0, rng.uniform(2, 6), canvas.shape)
    bgr = np.stack([canvas * rng.uniform(0.9, 1.0), canvas * rng.uniform(0.95, 1.0), canvas], 2)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(60, 90))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), (x0, y0, x1, y1)


def fake_page(word_imgs, rng=None, width=1400, line_gap=1.6, word_gap=(0.45, 0.8), margin=60, paper=245):
    """Paste word images (list of lines, each a list of uint8 gray words) onto a white page.
    Returns (page uint8 gray, boxes per line [(x0, y0, x1, y1), ...] of the ink)."""
    rng = rng or np.random.default_rng(0)
    hs = [w.shape[0] for line in word_imgs for w in line]
    lh = int(np.median(hs) * line_gap)
    H = margin * 2 + lh * len(word_imgs) + max(hs)
    page = np.full((H, width), paper, np.uint8)
    boxes, y = [], margin
    for line in word_imgs:
        x, row = margin, []
        for w in line:
            h, ww = w.shape
            if x + ww > width - margin:
                break
            yy = y + (lh - h) // 2
            region = page[yy:yy + h, x:x + ww]
            page[yy:yy + h, x:x + ww] = np.minimum(region, w)
            m = w < 180
            if m.any():
                ys, xs = np.nonzero(m)
                row.append((x + xs.min(), yy + ys.min(), x + xs.max() + 1, yy + ys.max() + 1))
            x += ww + int(rng.uniform(*word_gap) * np.median(hs))
        boxes.append(row)
        y += lh
    return page, boxes


def page_on_desk(page, rng=None, frame_hw=(720, 1280), tilt=0.03, scale=0.8, shift=(0, 0)):
    """The page photographed lying on a darker desk (slight perspective), as a BGR frame."""
    import cv2
    rng = rng or np.random.default_rng(0)
    H, W = frame_hw
    s = scale * min(W / page.shape[1], H / page.shape[0])
    pw, ph = page.shape[1] * s, page.shape[0] * s
    x0, y0 = (W - pw) / 2 + shift[0], (H - ph) / 2 + shift[1]
    dst = np.float32([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]])
    dst += rng.uniform(-tilt, tilt, (4, 2)).astype(np.float32) * [pw, ph]
    src = np.float32([[0, 0], [page.shape[1], 0], [page.shape[1], page.shape[0]], [0, page.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    desk = np.full((H, W), 70, np.uint8)
    warped = cv2.warpPerspective(page, M, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full(page.shape, 255, np.uint8), M, (W, H)) > 0
    out = np.where(mask, warped, desk).astype(np.float32)
    out = cv2.GaussianBlur(out, (0, 0), 0.8) + rng.normal(0, 3, out.shape)
    g = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), M

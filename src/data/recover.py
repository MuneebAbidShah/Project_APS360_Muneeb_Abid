"""Recover IAM labels from the metadata embedded in the original IAM PNG files.

Every original IAM word PNG carries its form's XML in a PNG text chunk (key 'xml'
or 'Description'). The XML holds the word's ID (in select="...@id='a01-000u-00-00'"),
its text, grammatical tag and bounding box, plus the line's segmentation status
(ok/err) and binarisation threshold. Those are exactly the fields of a words.txt
line, so a missing words.txt line can be rebuilt from the image alone:

    a01-000u-00-00 ok 154 408 768 27 51 AT A
    id             |  |   x   y   w  h  tag text
                   |  threshold
                   line segmentation status

Used by scripts/recover_labels.py. Sources of PNGs: the user's own images (if they
still carry the XML), or the Hugging Face copy of IAM (priyank-m/IAM_words_text_recognition),
whose PNGs are the originals with the XML intact.
"""
import hashlib
import io
import re
import xml.etree.ElementTree as ET

from PIL import Image

_SELECT_RE = re.compile(r"@id=(?:'|&apos;)([a-z]\d\d-\d{3}[a-z]?-\d\d-\d\d)(?:'|&apos;)")
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>")


def metadata_xml(img: Image.Image) -> str | None:
    """The embedded IAM XML of an opened PNG, or None."""
    for key in ("xml", "Description", "XML:com.adobe.xmp", "Comment"):
        v = img.info.get(key)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        if v and "<form" in v:
            return v
    return None


def entry_from_xml(xml: str) -> dict | None:
    """Parse one image's XML into the fields of its words.txt line."""
    m = _SELECT_RE.search(xml)
    if not m:
        return None
    wid = m.group(1)
    try:
        root = ET.fromstring(_DOCTYPE_RE.sub("", xml))
    except ET.ParseError:
        return None
    for line in root.iter("line"):
        for w in line.iter("word"):
            if w.get("id") != wid:
                continue
            cmps = w.findall("cmp")
            if cmps:
                xs = [int(c.get("x")) for c in cmps]; ys = [int(c.get("y")) for c in cmps]
                x2 = [int(c.get("x")) + int(c.get("width")) for c in cmps]
                y2 = [int(c.get("y")) + int(c.get("height")) for c in cmps]
                box = (min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys))
            else:
                box = (-1, -1, -1, -1)
            return {"id": wid, "status": line.get("segmentation", "ok"),
                    "threshold": line.get("threshold", "0"), "box": box,
                    "tag": w.get("tag", "UNK"), "text": w.get("text", ""),
                    "writer": root.get("writer-id")}
    return None


def entry_from_png_bytes(data: bytes) -> dict | None:
    with Image.open(io.BytesIO(data)) as im:   # text chunks are read without decoding pixels
        xml = metadata_xml(im)
    return entry_from_xml(xml) if xml else None


def entry_from_png_path(path) -> dict | None:
    try:
        with Image.open(path) as im:
            xml = metadata_xml(im)
        return entry_from_xml(xml) if xml else None
    except Exception:
        return None


def words_txt_line(e: dict) -> str:
    x, y, w, h = e["box"]
    return f"{e['id']} {e['status']} {e['threshold']} {x} {y} {w} {h} {e['tag']} {e['text']}"


def pixel_hash(img: Image.Image) -> str:
    """Hash of the decoded grayscale pixels + size: equal for identical images even if
    the PNG was re-compressed or its metadata stripped."""
    g = img.convert("L")
    return hashlib.sha1(f"{g.size}".encode() + g.tobytes()).hexdigest()

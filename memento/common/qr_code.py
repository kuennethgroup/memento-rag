"""memento/common/qr_code.py: is a figure image a QR code?

Journal first pages often carry a QR code linking to the article; MinerU saves it as an image and enrichment would give
it the nearest paragraph as its text, so it could be cited and shown as a figure. A QR code is recognised by decoding it
(OpenCV): only an image that decodes counts, so a blocky chart or micrograph is never taken for one. Only roughly square
images are checked, which keeps the cost to a few milliseconds. Without OpenCV, or for an unreadable file: not a QR
code."""
from pathlib import Path


def is_qr_code(path) -> bool:
    if not path:
        return False
    try:
        from PIL import Image
        with Image.open(Path(path)) as im:
            w, h = im.size
        if not h or not 0.8 <= w / h <= 1.25:
            return False
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return False
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        return bool(data)
    except Exception:
        return False

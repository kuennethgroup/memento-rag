# Journal QR codes are no figures: a QR code on a paper's first page (a link to its DOI) is not indexed as a figure.
import sys
from pathlib import Path

import cv2
import numpy as np
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.common.qr_code import is_qr_code  # noqa: E402


def qr_png(path: Path, text="http://doi.wiley.com/10.1002/aelm.201600505", scale=5):
    code = cv2.QRCodeEncoder.create().encode(text)
    img = cv2.resize(code, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255))
    return path


def chart_png(path: Path):
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 180), (110, 280), (200, 120, 40), -1)
    cv2.rectangle(img, (170, 60), (220, 280), (200, 120, 40), -1)
    cv2.line(img, (40, 280), (280, 280), (0, 0, 0), 2)
    cv2.imwrite(str(path), img)
    return path


def test_qr_code_detected(tmp_path):
    assert is_qr_code(qr_png(tmp_path / "qr.png"))


def test_figures_are_not_qr_codes(tmp_path):
    assert not is_qr_code(chart_png(tmp_path / "chart.png"))
    wide = np.full((200, 800, 3), 255, np.uint8)                       # a wide figure is never checked
    cv2.imwrite(str(tmp_path / "wide.png"), wide)
    assert not is_qr_code(tmp_path / "wide.png")


def test_unreadable_is_not_a_qr_code(tmp_path):
    assert not is_qr_code(tmp_path / "missing.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    assert not is_qr_code(tmp_path / "broken.png")
    assert not is_qr_code(None)

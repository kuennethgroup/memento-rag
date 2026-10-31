# The shared cloud quota from rate-limit headers.
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from memento.config.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from memento.common import cloud_quota as cq  # noqa: E402

H = {"X-RateLimit-Remaining-Minute": "29", "X-RateLimit-Remaining-Hour": "187", "X-RateLimit-Remaining-Day": "935",
     "X-RateLimit-Remaining-Month": "2935", "X-RateLimit-Limit-Minute": "30", "X-RateLimit-Limit-Hour": "200",
     "X-RateLimit-Limit-Day": "1000", "X-RateLimit-Limit-Month": "3000", "RateLimit-Reset": "23"}
T = datetime(2026, 9, 29, 13, 50, 10, tzinfo=timezone.utc)


def test_headers_become_state_with_resets():
    s = cq.from_headers({k.lower(): v for k, v in H.items()}, T)          # header names are case-insensitive
    assert s["windows"]["hour"] == {"remaining": 187, "limit": 200, "reset_at": "2026-09-29T14:00:00Z"}
    assert s["windows"]["minute"]["reset_at"] == "2026-09-29T13:50:33Z"
    assert s["windows"]["day"]["reset_at"] == "2026-09-30T00:00:00Z"
    assert s["windows"]["month"]["reset_at"] == "2026-10-01T00:00:00Z"
    assert cq.from_headers({"content-type": "text/plain"}, T) is None     # no rate-limit headers, no reading


def test_december_rolls_over_to_january():
    s = cq.from_headers(H, datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc))
    assert s["windows"]["month"]["reset_at"] == "2027-01-01T00:00:00Z"


def test_available_until_the_reserve():
    s = cq.from_headers(H, T)
    assert cq.availability(s, T) == {"available": True, "blocked_by": None, "back_at": None, "left_hour": 187}
    s["windows"]["hour"]["remaining"] = 3
    a = cq.availability(s, T)
    assert a["available"] is False and a["blocked_by"] == "hour" and a["back_at"] == "2026-09-29T14:00:00Z"


def test_minute_limit_never_blocks():
    s = cq.from_headers(H, T)
    s["windows"]["minute"]["remaining"] = 0
    assert cq.availability(s, T)["available"] is True


def test_reading_before_reset_counts_as_full():
    s = cq.from_headers(H, T)
    s["windows"]["hour"]["remaining"] = 0
    a = cq.availability(s, datetime(2026, 9, 29, 14, 5, tzinfo=timezone.utc))
    assert a["available"] is True and a["left_hour"] == 200


def test_day_and_month_block_too():
    s = cq.from_headers(H, T)
    s["windows"]["day"]["remaining"] = 1
    assert cq.availability(s, T)["blocked_by"] == "day"
    s["windows"]["month"]["remaining"] = 0
    assert cq.availability(s, T)["blocked_by"] == "month"                 # the longest wait is reported


def test_state_file_roundtrip_is_atomic(tmp_path):
    p = cq.state_path(tmp_path)
    cq.write_state(p, cq.from_headers(H, T), ["memento-query-qwen35"])
    data = json.loads(p.read_text())
    assert data["cloud_models"] == ["memento-query-qwen35"] and data["reserve"] == cq.RESERVE
    assert not list(p.parent.glob("*.tmp"))
    assert cq.read_state(p) == data


def test_read_state_missing_or_corrupt(tmp_path):
    assert cq.read_state(tmp_path / "none.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cq.read_state(bad) is None
    assert cq.availability(None, T)["available"] is True

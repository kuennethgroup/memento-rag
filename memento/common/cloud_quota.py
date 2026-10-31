# memento/common/cloud_quota.py: the shared cloud-model quota, from the rate-limit headers of each reply
"""The cloud gateway reports the key's remaining calls per minute, hour, day and month on every reply. Hour, day and
month are fixed clock windows in UTC; a reading taken before a window's reset counts as a full window afterwards."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

WINDOWS = ("minute", "hour", "day", "month")
RESERVE = 3          # calls kept back for questions already on their way


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _reset(window: str, now: datetime, minute_reset_s: int) -> datetime:
    now = now.astimezone(timezone.utc)
    if window == "minute":
        return now + timedelta(seconds=minute_reset_s)
    if window == "hour":
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "day":
        return midnight + timedelta(days=1)
    first = midnight.replace(day=1)
    return first.replace(year=first.year + 1, month=1) if first.month == 12 else first.replace(month=first.month + 1)


def from_headers(headers, now: datetime) -> dict | None:
    """A quota reading from a reply's headers, or None when the reply carries none."""
    h = {k.lower(): v for k, v in dict(headers).items()}
    if "x-ratelimit-remaining-hour" not in h:
        return None
    minute_reset = int(h.get("ratelimit-reset") or 60)
    windows = {w: {"remaining": int(h.get(f"x-ratelimit-remaining-{w}", 0)),
                   "limit": int(h.get(f"x-ratelimit-limit-{w}", 0)),
                   "reset_at": _iso(_reset(w, now, minute_reset))} for w in WINDOWS}
    return {"at": _iso(now), "reserve": RESERVE, "windows": windows}


def availability(state: dict | None, now: datetime) -> dict:
    """Cloud models usable now? Not while the hour, day or month window (not yet reset) is at or below the reserve.
    No reading at all counts as available: the answer path still falls back to the local model on a 429."""
    if not state or not state.get("windows"):
        return {"available": True, "blocked_by": None, "back_at": None, "left_hour": None}
    left = {w: (win["limit"] if now >= _parse(win["reset_at"]) else win["remaining"])
            for w, win in state["windows"].items()}
    for w in ("month", "day", "hour"):
        if w in left and left[w] <= state.get("reserve", RESERVE):
            return {"available": False, "blocked_by": w, "back_at": state["windows"][w]["reset_at"],
                    "left_hour": left.get("hour")}
    return {"available": True, "blocked_by": None, "back_at": None, "left_hour": left.get("hour")}


def state_path(output_root) -> Path:
    return Path(output_root) / "_runtime" / "cloud_quota.json"


def write_state(path: Path, state: dict, cloud_models: list[str]) -> None:
    """Counts and times only, written atomically (temp file, then rename) so a reader never sees half a file."""
    from memento.common.atomic_file import write_text_atomic
    # a temporary name per process and thread: copies of the pipes can run in one process
    write_text_atomic(path, json.dumps({**state, "cloud_models": cloud_models}))


def read_state(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None

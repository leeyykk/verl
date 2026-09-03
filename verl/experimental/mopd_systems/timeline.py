"""Opt-in JSONL timeline events for MOPD systems probes.

Set ``VERL_MOPD_TIMELINE_JSONL=/path/to/timeline.jsonl`` to enable. The helper
is intentionally tiny and dependency-free so it can be called from Ray workers
without changing normal verl behavior.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def emit_timeline_event(event: str, **payload: Any) -> None:
    path = os.environ.get("VERL_MOPD_TIMELINE_JSONL")
    if not path:
        return

    record = {
        "event": event,
        "time_s": time.time(),
        "time_ns": time.time_ns(),
        "pid": os.getpid(),
        **{key: _jsonable(value) for key, value in payload.items()},
    }
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with _LOCK:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception:
        # Timeline emission must never perturb a training run.
        return

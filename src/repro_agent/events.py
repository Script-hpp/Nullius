"""Ereignisprotokoll eines Laufs (`runs/<paper>/<ts>/events.jsonl`). Grundlage der Evidence-Ansicht (web/).

Eine JSON-Zeile pro Ereignis: {"ts": Unix-Zeit, "type": ..., ...Felder}. Nur Tatsachen des Laufs, keine Schluesse.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, type: str, **data) -> None:
        if self.path is None:
            return
        line = json.dumps({"ts": time.time(), "type": type, **data}, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")

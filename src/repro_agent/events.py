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

    def write(self, type: str, **data) -> dict:
        """Schreibt die Zeile (falls ein Pfad gesetzt ist) und gibt den Record zurueck."""
        record = {"ts": time.time(), "type": type, **data}
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a") as f:
                f.write(line + "\n")
        return record

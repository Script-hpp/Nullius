"""Zeitlimit je Seed, abgeleitet aus der Laufzeit der Referenzloesung auf demselben Backend (generisch, ohne Paper-Wissen).

Ein fester Wert (TRAIN_TIMEOUT_S = 600 s) passt auf Nebius, aber nicht auf jeden Rechner oder jedes Modell: Braucht die (effiziente)
Referenzloesung auf dem lokalen Docker-Backend schon 100 s, laeuft ein Agenten-Code, der ein Mehrfaches davon braucht, nie unter 600 s.
Das Limit ist Faktor * Referenzlaufzeit (der Faktor 10 laesst dem Agenten-Code eine Grossenordnung Luft gegenueber der Referenz), nie
unter dem Standard, nie ueber dem Deckel; es gilt fuer alle Arme und Proben einer Einheit gleich. Ist die Referenz schnell (unter
60 s), bleibt es bei 600 s.
"""

from __future__ import annotations

TIMEOUT_FACTOR = 10.0
TIMEOUT_CAP_S = 3600


def train_timeout(ref_seconds: float, default: int = 600, factor: float = TIMEOUT_FACTOR, cap: int = TIMEOUT_CAP_S) -> int:
    """max(default, factor * ref_seconds), hoechstens cap; aufgerundet auf volle 60 s."""
    t = max(float(default), factor * float(ref_seconds))
    return int(min(cap, -(-t // 60) * 60))

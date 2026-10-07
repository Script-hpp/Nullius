"""Statische Pruefung von Agenten-Code auf Netzzugriff/Downloads/Prozesse.

Zweite Schicht neben dem abgeschalteten Netz im Trainingslauf (`unshare -n`). Wortlisten sind
umgehbar, deshalb nie allein verlassen. Treffer machen den Versuch `ungueltig`.
"""

from __future__ import annotations

import re

# (Regex, Beschreibung). `(?<![.\w])` verhindert Treffer auf model.eval(), self.exec(...) usw.
FORBIDDEN: list[tuple[str, str]] = [
    (r"://", "URL im Code"),
    (r"\b(urllib|requests|httpx|aiohttp|http\.client|ftplib|socket|paramiko|smtplib)\b", "Netzwerk-Modul"),
    (r"\b(subprocess|pexpect)\b|\bos\.(system|popen|exec\w*|spawn\w*)\b", "Prozessstart"),
    (r"(?<![.\w])(exec|eval|compile)\s*\(", "dynamische Codeausfuehrung"),
    (r"__import__|importlib", "dynamischer Import"),
    (r"\b(torch_geometric|torchvision\.datasets|torchtext|tensorflow_datasets|ogb|dgl)\b", "Dataset-Bibliothek"),
    (r"\bfetch_\w+|\bPlanetoid\b|\bdownload\w*", "Download-Funktion"),
    (r"\b(pip|wget|curl|git)\b", "Installations-/Download-Tool"),
    (r"\bind\.(cora|citeseer|pubmed)\b|planetoid", "Zugriff auf Rohdaten des Datensatzes"),
]


def find_violations(code: str) -> list[str]:
    hits = []
    for pattern, desc in FORBIDDEN:
        m = re.search(pattern, code)
        if m:
            line = code.count("\n", 0, m.start()) + 1
            hits.append(f"{desc}: '{m.group(0)}' in Zeile {line}")
    return hits

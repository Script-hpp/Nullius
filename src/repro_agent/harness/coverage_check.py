"""Ausfuehrungsabdeckung der vom Pruefer zitierten Codestellen. Deterministisch, kein LLM.

Grenze (wichtig): Zeilenabdeckung zeigt nur, ob eine Zeile je ausgefuehrt wurde. Code, der ausgefuehrt, dessen
Ergebnis aber verworfen wird (`trend = decomp(x)` und dann ungenutzt; Adjazenz gebaut und nie multipliziert),
gilt als abgedeckt. Solche Faelle erkennt erst ein Identitaets-/Ablationstest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX = re.compile(r"^\s*\d+\|\s?", flags=re.M)   # Zeilennummern-Format des Prueferprompts


def _n(s: str) -> str:
    return re.sub(r"\s+", "", s)


def locate_quote(code: str, quote: str) -> tuple[int, int] | None:
    """1-basierte Zeilen (von, bis) der ersten Fundstelle. Leerraum und Leerzeilen sind egal."""
    code_lines = [(i, _n(line)) for i, line in enumerate(code.splitlines(), 1) if _n(line)]
    q = [_n(line) for line in _PREFIX.sub("", quote).splitlines() if _n(line)]
    if not q:
        return None
    for j in range(len(code_lines) - len(q) + 1):
        if [c for _, c in code_lines[j:j + len(q)]] == q:
            return code_lines[j][0], code_lines[j + len(q) - 1][0]
    if len(q) == 1:   # Zitat ist nur ein Teil einer Zeile
        for i, c in code_lines:
            if q[0] in c:
                return i, i
    return None


@dataclass(frozen=True)
class Coverage:
    executed: frozenset[int]
    missing: frozenset[int]   # ausfuehrbare, nie ausgefuehrte Zeilen

    @staticmethod
    def from_coverage_json(data: dict, filename: str = "train.py") -> Coverage | None:
        f = next((v for k, v in data.get("files", {}).items() if k.endswith(filename)), None)
        if f is None:
            return None
        return Coverage(frozenset(f["executed_lines"]), frozenset(f["missing_lines"]))


def quote_coverage(code: str, quote: str, cov: Coverage) -> str:
    """ausgefuehrt | teilweise ausgefuehrt | nicht ausgefuehrt | nicht pruefbar (Zitat nicht gefunden, keine Codezeile)."""
    span = locate_quote(code, quote)
    if span is None:
        return "nicht pruefbar"
    lines = range(span[0], span[1] + 1)
    ran = sum(i in cov.executed for i in lines)
    not_ran = sum(i in cov.missing for i in lines)
    if ran + not_ran == 0:
        return "nicht pruefbar"
    if ran == 0:
        return "nicht ausgefuehrt"
    return "ausgefuehrt" if not_ran == 0 else "teilweise ausgefuehrt"

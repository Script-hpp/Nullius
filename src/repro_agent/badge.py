"""Abzeichen (SVG) je exportiertem Lauf, fuer das README der Autoren: `Nullius | reproduced with our setup`, darunter gemessen gegen behauptet.

Text nur aus dem Dossier. Dieselben Urteilsworte wie die Kopfzeile des Zeugnisses (web/src/labels.ts, RESULT_CATALOG und catalogLabel), Achse 1 bleibt
sichtbar. Ist die Methodentreue nicht geprueft, steht "method not checked" im Abzeichen; weicht die Gesamtaussage ab ("Zielwert erreicht mit anderem
Modell"), steht sie zusaetzlich. Keine Zahl ohne "measured"/"claimed". Sprache Englisch (Ziel: README der Autoren).
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from .dossier import Dossier

# Wortlaut wie web/src/labels.ts RESULT_CATALOG (Kopfzeile des Zeugnisses), klein geschrieben
WORDS = {
    "reproduziert": ("reproduced", "green"),
    "teilweise": ("partially reproduced", "amber"),
    "nicht reproduziert": ("not reproduced", "red"),
    "nicht reproduzierbar (Setup)": ("setup not reproducible", "gray"),
    "Agenten-Fehler vermutet": ("agent error suspected", "violet"),
}
NOT_ASSESSABLE = "nicht bewertbar (Methode nicht umgesetzt)"
OVERALL = {   # nur Gesamtaussagen, die ueber Achse 1 hinaus etwas sagen (web/src/labels.ts OVERALL)
    "Zielwert erreicht mit anderem Modell": "overall: target reached with a different model",
    "Paper reproduziert": "overall: paper reproduced (target value and method)",
}
# Solide Toenung, Weiss darauf lesbar (>= 4,5:1); folgt den Statusfarben der App (gruen, bernstein, rot, violett, grau)
COLORS = {"green": "#047857", "amber": "#a16207", "red": "#be123c", "violet": "#6d28d9", "gray": "#52525b"}
LEFT = "#18181b"
MUTED = "#71717a"   # lesbar auf hellem und dunklem README-Hintergrund
CHAR_W = 6.8        # Verdana 11 px, vorsichtig geschaetzt (lieber etwas Luft als abgeschnittener Text); kein textLength, das verzerrt je nach Schrift


def _value(metric: str, x: float) -> str:
    return f"{x * 100:.1f}%" if metric == "accuracy" else f"{x:.3f}"


def _date(timestamp: str) -> str | None:
    m = re.match(r"(\d{4})(\d{2})(\d{2})-", timestamp)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else None


def badge_lines(dossier: Dossier, timestamp: str, catalog: bool = False) -> dict | None:
    """Texte und Farbe des Abzeichens; None, wenn kein Urteil mit Messwert da ist."""
    if not dossier.verdicts or not dossier.claims:
        return None
    v = dossier.verdicts[0]
    claim = next((c for c in dossier.claims if c.id == v.claim_id), dossier.claims[0])
    gesamt = v.gesamtaussage
    if catalog and gesamt == NOT_ASSESSABLE:
        word, tone = "not assessable", "gray"
    else:
        word, tone = WORDS.get(v.urteil, (v.urteil, "gray"))
    has_value = v.measured_mean == v.measured_mean and v.urteil in ("reproduziert", "teilweise", "nicht reproduziert")
    if word == "not assessable":   # wie das Zeugnis: die Messung sagt nichts ueber die Behauptung
        nums = f"method not implemented · claimed {_value(claim.metric, claim.target)}"
    else:
        nums = f"measured {_value(claim.metric, v.measured_mean)} · claimed {_value(claim.metric, claim.target)}" if has_value else f"claimed {_value(claim.metric, claim.target)}"
    extra = []
    if word == "not assessable":
        pass
    elif v.methode == "nicht geprueft":
        extra.append("method not checked")
    elif gesamt in OVERALL:
        extra.append(OVERALL[gesamt])
    return {
        "label": "Nullius", "verdict": f"{word} with our setup", "tone": tone,
        "detail": " · ".join([nums, *extra]), "date": _date(timestamp),
    }


def render_badge(lines: dict) -> str:
    pad = 8
    wl = len(lines["label"]) * CHAR_W + 2 * pad
    wv = len(lines["verdict"]) * CHAR_W + 2 * pad
    width = max(wl + wv, len(lines["detail"]) * 6.2 + 4)
    height = 52 if lines["date"] else 40
    date = f'<text x="2" y="50" font-size="9" fill="{MUTED}">{escape(lines["date"])}</text>' if lines["date"] else ""
    title = escape(f'{lines["label"]}: {lines["verdict"]}. {lines["detail"]}.' + (f' Run of {lines["date"]}.' if lines["date"] else ""), {'"': "&quot;"})   # Attributwert: Anfuehrungszeichen escapen
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height}" viewBox="0 0 {width:.0f} {height}" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        f'<g font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
        f'<clipPath id="r"><rect width="{wl + wv:.0f}" height="22" rx="4"/></clipPath>'
        f'<g clip-path="url(#r)"><rect width="{wl:.0f}" height="22" fill="{LEFT}"/><rect x="{wl:.0f}" width="{wv:.0f}" height="22" fill="{COLORS[lines["tone"]]}"/></g>'
        f'<text x="{pad}" y="15" fill="#fff" font-weight="bold">{escape(lines["label"])}</text>'
        f'<text x="{wl + pad:.0f}" y="15" fill="#fff">{escape(lines["verdict"])}</text>'
        f'<text x="2" y="36" font-size="10" fill="{MUTED}">{escape(lines["detail"])}</text>'
        f"{date}</g></svg>\n"
    )


def write_badge(dest: Path, dossier: Dossier, timestamp: str, catalog: bool = False) -> Path | None:
    """Schreibt dest/badge.svg; kein Urteil -> keine Datei (alte Datei wird entfernt)."""
    lines = badge_lines(dossier, timestamp, catalog)
    path = dest / "badge.svg"
    if lines is None:
        path.unlink(missing_ok=True)
        return None
    path.write_text(render_badge(lines))
    return path

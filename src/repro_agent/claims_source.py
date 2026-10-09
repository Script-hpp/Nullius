"""Tavily-Rolle 2: Zielwert aus dem Paper lesen -- und deterministisch gegenpruefen.

Prinzip wie beim Harness: die KI liest, ein Programm prueft. Ein Zielwert wird nur uebernommen,
wenn der woertliche Textausschnitt UND die Zahl im extrahierten Quelltext vorkommen.

Quellen-Kette: arXiv-HTML (nur neue Papers) -> ar5iv -> PDF. Jede Antwort wird mit URL,
Zeitstempel und Hash als Fixture gespeichert (Beleg im Zeugnis + Testdaten, spart Credits).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .llm import LLMClient
from .table_check import check_columns
from .tavily_tools import TavilyTools

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures"
ARXIV_HTML_MIN_ID = "2312"  # arXiv-HTML gibt es erst fuer Einreichungen ab ca. Ende 2023

EXTRACT_SYSTEM = (
    "You read a paper's text and report ONE number exactly as printed. Reply with a single JSON "
    "object and nothing else. IMPORTANT: You must double-escape all LaTeX backslashes in your JSON strings "
    "(e.g. use \\\\textbf instead of \\textbf)."
)


@dataclass
class PaperText:
    arxiv_id: str
    url: str
    source_kind: str            # arxiv_html | ar5iv | pdf
    retrieved_at: str
    sha256: str
    arxiv_version: str | None   # Version, deren URL abgerufen wurde (None = unversioniert, z. B. ar5iv)
    text: str


@dataclass
class TargetEvidence:
    value: float                # als Bruch, z. B. 0.815
    quote: str
    table: str | None
    url: str
    source_kind: str
    retrieved_at: str
    sha256: str
    arxiv_version: str | None
    caveat: str | None = None   # z. B. Spaltenzuordnung nicht belegt
    column_check: str = "nicht pruefbar"   # "ok" (Kopfzeile geparst, Position stimmt) | "nicht pruefbar"
    from_store: bool = False    # True: aus gespeicherter Extraktion (Fixture) gelesen, kein neuer Modellaufruf


class ExtractionInvalid(RuntimeError):
    pass


def _source_urls(arxiv_id: str, version: str | None) -> list[tuple[str, str]]:
    """Versionierte URLs (z. B. .../pdf/1609.02907v4), damit Text und Version belegt zusammenpassen.
    ar5iv kennt keine Versionen -> dort bleibt die Version unbelegt (steht so im Beleg)."""
    v = version or ""
    urls = []
    if arxiv_id[:4] >= ARXIV_HTML_MIN_ID:
        urls.append(("arxiv_html", f"https://arxiv.org/html/{arxiv_id}{v}"))
    if version:
        urls.append(("pdf", f"https://arxiv.org/pdf/{arxiv_id}{v}"))
    urls.append(("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"))  # unversioniert: letzter Ausweg
    if not version:
        urls.append(("pdf", f"https://arxiv.org/pdf/{arxiv_id}"))
    return urls


def latest_arxiv_version(arxiv_id: str) -> str | None:
    """Kostenlos ueber die arXiv-API (kein Tavily-Credit)."""
    try:
        r = httpx.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id}, timeout=20)
        m = re.search(r"arxiv\.org/abs/[\w.]+?(v\d+)</id>", r.text)
        return m.group(1) if m else None
    except httpx.HTTPError:
        return None


def fetch_paper_text(
    arxiv_id: str, get_tavily: Callable[[], TavilyTools], fixture_dir: Path | None = None, mode: str = "fixture"
) -> PaperText:
    """mode="fixture": Fixture zuerst, `get_tavily` nur bei Fixture-Miss (spart Credits); bei Miss wird die Fixture angelegt.
    mode="live": immer Tavily-Aufruf (Standard des Orchestrators); schreibt nie nach `fixture_dir`
    (das Ergebnis steht im Lauf: paper_text.json, Dossier.sources)."""
    fixture_dir = fixture_dir or FIXTURE_DIR
    fixture = fixture_dir / f"{arxiv_id}.json"
    if mode == "fixture" and fixture.exists():
        return PaperText(**json.loads(fixture.read_text()))
    tavily = get_tavily()

    version = latest_arxiv_version(arxiv_id)
    
    # Neu: Zuerst versuchen wir, den echten LaTeX-Quellcode von arXiv zu holen.
    from .discovery import _fetch_arxiv_source
    latex_text = _fetch_arxiv_source(arxiv_id)
    if latex_text and len(latex_text) > 2000:
        paper = PaperText(
            arxiv_id=arxiv_id, url=f"https://export.arxiv.org/e-print/{arxiv_id}", source_kind="latex",
            retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sha256=hashlib.sha256(latex_text.encode()).hexdigest(),
            arxiv_version=version, text=latex_text,
        )
        if mode != "live":
            fixture_dir.mkdir(parents=True, exist_ok=True)
            fixture.write_text(json.dumps(asdict(paper), ensure_ascii=False, indent=1))
        return paper

    # Fallback: Tavily (liest oft unsauber, mischt Tabellenspalten)
    for kind, url in _source_urls(arxiv_id, version):
        result = tavily.read_claims(url)
        for item in result.get("results", []):
            text = item.get("raw_content") or ""
            if len(text) > 2000:  # leere/Fehlerseiten aussortieren
                paper = PaperText(
                    arxiv_id=arxiv_id, url=url, source_kind=kind,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    arxiv_version=version if kind != "ar5iv" else None, text=text,
                )
                if mode != "live":
                    fixture_dir.mkdir(parents=True, exist_ok=True)
                    fixture.write_text(json.dumps(asdict(paper), ensure_ascii=False, indent=1))
                return paper
    raise ExtractionInvalid(f"Kein Volltext fuer {arxiv_id} ueber {[k for k, _ in _source_urls(arxiv_id, version)]}")


def _norm_v1(s: str) -> str:
    """Alte Normalisierung (Stand vor Tavily v2), nur fuer Vorher/Nachher-Vergleiche (scripts/measure_role4_drops.py)."""
    s = re.sub(r"\s+", " ", s.replace(chr(0xA0), " ")).strip().lower()
    s = re.sub(r"(\d) \.(\d)", r"\1.\2", s)
    return re.sub(r"(?<=[a-z])-\s?(?=[a-z])", "", s)


def clean_latex_quote(text: str) -> str:
    """Entfernt grundlegende LaTeX-Befehle fuer die saubere Anzeige in Katalog und Zeugnis."""
    if not text: return ""
    text = re.sub(r'\\(?:cellcolor|color|textcolor|rowcolor)(?:\[.*?\])?(?:\{.*?\})?', '', text)
    while re.search(r'\\(?:textbf|textit|underline|mathit)\{([^{}]*)\}', text):
        text = re.sub(r'\\(?:textbf|textit|underline|mathit)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\pm\b', '±', text)
    text = re.sub(r'\\[a-zA-Z]+(?:\[.*?\])?', '', text)
    text = re.sub(r'\$|&|\\\\', ' ', text)
    text = re.sub(r'\{|\}', '', text)
    return re.sub(r'\s+', ' ', text).strip()


_TYPO = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
})


def _norm(s: str) -> str:
    """Text fuer den Zitatvergleich glaetten (Tavily v2). Nur Schreibweise, keine Zahlen und Bezeichner:
    NFKC (Ligaturen fi/fl), typografische Striche und Anfuehrungszeichen, weiche Trennstriche, Leerraum (auch vor
    Satzzeichen), PDF-Zahlen '81 .5' -> '81.5', Silbentrennung am Zeilenende."""
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "").translate(_TYPO)
    s = re.sub(r"\s+", " ", s.replace(chr(0xA0), " ")).strip().lower()
    s = re.sub(r"(\d) \.(\d)", r"\1.\2", s)
    s = re.sub(r"\s+([.,;:!?)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return re.sub(r"(?<=[a-z])-\s?(?=[a-z])", "", s)   # Silbentrennung der PDF-Extraktion ("prop-agation")


def _windows(text: str, keyword: str, width: int = 2500, max_chars: int = 14000, prefer: str | None = None) -> str:
    """Nur Textstellen um das Stichwort an das Modell geben (Tabellen), nicht das ganze Paper.

    Passt mehr Text um das Stichwort, als `max_chars` erlaubt (Datensatzname steht im ganzen Paper), wuerde der Anfang
    des Papers die Tabelle verdraengen (GAT: Tabelle 2 stand hinter der Grenze). Mit `prefer` (Methodenname) kommen
    zuerst die Fenster, die ihn enthalten und viele Dezimalzahlen tragen; die Reihenfolge im Text bleibt erhalten."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(re.escape(keyword), text, flags=re.I):
        lo, hi = max(0, m.start() - width), min(len(text), m.end() + width)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
    if prefer:
        def score(span: tuple[int, int]) -> tuple[int, int]:
            chunk = text[span[0]:span[1]]
            return (int(prefer.lower() in chunk.lower()), len(re.findall(r"\d+ ?\.\d", chunk)))

        chosen, used = [], 0
        for span in sorted(spans, key=score, reverse=True):
            if used and used + (span[1] - span[0]) > max_chars:
                continue
            chosen.append(span)
            used += span[1] - span[0]
        spans = sorted(chosen)
    return "\n...\n".join(text[a:b] for a, b in spans)[:max_chars]


def verify_extraction(data: dict, paper_text: str, percent: bool = True) -> float:
    """Deterministischer Check. Gibt den Wert als Bruch (0..1) zurueck oder wirft ExtractionInvalid."""
    quote, number = str(data.get("quote", "")), data.get("value")
    if not quote or number is None:
        raise ExtractionInvalid("Modell lieferte keinen Textausschnitt oder keine Zahl")
    if _norm(quote) not in _norm(paper_text):
        raise ExtractionInvalid(f"Textausschnitt nicht im Quelltext: {quote!r}")
    try:
        value = float(number)
    except (TypeError, ValueError) as exc:
        raise ExtractionInvalid(f"Zahl nicht lesbar: {number!r}") from exc
    printed = re.findall(r"\d+(?:[.,]\d+)?", _norm(quote))
    if not any(abs(float(p.replace(",", ".")) - value) < 1e-9 for p in printed):
        raise ExtractionInvalid(f"Zahl {value} steht nicht im Textausschnitt {quote!r}")
    return value / 100 if percent and value > 1 else value  # Prozentangabe -> Bruch (Accuracy)


def _last_json_object(content: str) -> dict | None:
    """Letztes gueltiges JSON-Objekt (Modelle schreiben Denkprozess davor)."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", content))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "quote" in value:
            return value
    return None


def _column_caveat(method: str, quote: str) -> str | None:
    """Rueckfall, wenn die Kopfzeile nicht geparst werden konnte."""
    if _norm(method.split(" (")[0]) in _norm(quote):
        return None
    return (
        f"Methodenname '{method}' steht nicht im Zitat: Der Check belegt nur, dass die Zahl im "
        "Text steht, nicht dass sie zur richtigen Spalte/Zeile gehoert (Zuordnung nicht automatisch "
        "geprueft, Kopfzeile ggf. von Hand gegenlesen)."
    )


def _column_status(
    text: str, quote: str, value: float, method: str, dataset: str, metric: str, percent: bool
) -> tuple[str | None, str]:
    """(caveat, column_check). Falsche Spalte = ExtractionInvalid, Zielwert ungueltig."""
    res = check_columns(_norm(text), _norm(quote), value, method, dataset, metric, percent)
    if res.status == "mismatch":
        raise ExtractionInvalid(f"Spaltenpruefung: {res.detail}")
    if res.status == "ok":
        return None, "ok"
    return _column_caveat(method, quote), "nicht pruefbar"


def extract_target(
    llm: LLMClient, paper: PaperText, method: str, dataset: str, metric: str, percent: bool = True
) -> TargetEvidence:
    context = _windows(paper.text, dataset, prefer=method)
    if not context:
        raise ExtractionInvalid(f"'{dataset}' kommt im Quelltext nicht vor")
    prompt = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": (
            f"Find the reported {metric} of method '{method}' on dataset '{dataset}'.\n"
            'JSON keys: "value" (number exactly as printed, e.g. 81.5), "table" (e.g. "Table 2"), '
            '"quote" (a VERBATIM excerpt copied from the text that contains the number, ideally the '
            "table row). Do not compute or guess; if not printed, use null for value. Keep your "
            "reasoning short and end with the JSON object.\n\n"
            f"TEXT:\n{context}"
        )},
    ]
    last: ExtractionInvalid | None = None
    max_tokens = 6000
    for attempt in range(3):  # Free-Modelle antworten gelegentlich ohne nutzbares JSON
        # Wiederholung mit anderer Temperatur (sonst kaeme dieselbe Antwort aus dem Cache) und bei Abschnitt doppelter Grenze
        resp = llm.complete("fast", prompt, temperature=0.1 * attempt, max_tokens=max_tokens, timeout_s=400)
        data = _last_json_object(resp.content)
        if data is None:
            last = ExtractionInvalid("Modellantwort enthaelt kein JSON")
            if resp.finish_reason == "length":
                max_tokens *= 2
            continue
        try:
            value = verify_extraction(data, paper.text, percent)
            _column_status(paper.text, str(data["quote"]), value, method, dataset, metric, percent)
            break
        except ExtractionInvalid as exc:
            last = exc
    else:
        raise last  # type: ignore[misc]
    quote = str(data["quote"])
    caveat, column_check = _column_status(paper.text, quote, value, method, dataset, metric, percent)
    return TargetEvidence(
        value=value, quote=quote, table=data.get("table"), url=paper.url,
        source_kind=paper.source_kind, retrieved_at=paper.retrieved_at, sha256=paper.sha256,
        arxiv_version=paper.arxiv_version, caveat=caveat, column_check=column_check,
    )


def extract_target_cached(
    llm: LLMClient, paper: PaperText, method: str, dataset: str, metric: str,
    fixture_dir: Path | None = None, percent: bool = True, mode: str = "fixture",
) -> TargetEvidence:
    """Verifizierten Zielwert pro (Paper-Text, Methode, Datensatz, Metrik) einmal speichern.

    Der Wert wird bei jedem Laden erneut gegen den Quelltext geprueft (Zitat + Zahl), der Cache
    ersetzt also nur den LLM-Aufruf, nicht die Pruefung. Spart Anfragen auf Free-Endpunkten.
    mode="live": liest vorhandene Eintraege, schreibt aber nie nach `fixture_dir`.
    """
    fixture_dir = fixture_dir or FIXTURE_DIR
    raw = f"{paper.sha256}|{method}|{dataset}|{metric}" + ("" if percent else "|raw")
    key = hashlib.sha256(raw.encode()).hexdigest()[:16]
    path = fixture_dir / f"{paper.arxiv_id}_target_{key}.json"
    if path.exists():
        ev = TargetEvidence(**{**json.loads(path.read_text()), "from_store": True})
        ev.caveat, ev.column_check = _column_status(   # immer neu berechnen, auch fuer alte Fixtures
            paper.text, ev.quote, ev.value, method, dataset, metric, percent
        )
        verify_extraction(
            {"value": ev.value * 100 if percent else ev.value, "quote": ev.quote}, paper.text, percent
        )
        return ev
    ev = extract_target(llm, paper, method, dataset, metric, percent)
    if mode == "live":
        return ev
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(ev), ensure_ascii=False, indent=1))
    return ev

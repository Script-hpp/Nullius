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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .llm import LLMClient
from .tavily_tools import TavilyTools

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "fixtures"
ARXIV_HTML_MIN_ID = "2312"  # arXiv-HTML gibt es erst fuer Einreichungen ab ca. Ende 2023

EXTRACT_SYSTEM = (
    "You read a paper's text and report ONE number exactly as printed. Reply with a single JSON "
    "object and nothing else."
)


@dataclass
class PaperText:
    arxiv_id: str
    url: str
    source_kind: str            # arxiv_html | ar5iv | pdf
    retrieved_at: str
    sha256: str
    arxiv_version: str | None   # neueste Version laut arXiv-API (Inhalt von ar5iv = irgendeine davon)
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


class ExtractionInvalid(RuntimeError):
    pass


def _source_urls(arxiv_id: str) -> list[tuple[str, str]]:
    urls = []
    if arxiv_id[:4] >= ARXIV_HTML_MIN_ID:
        urls.append(("arxiv_html", f"https://arxiv.org/html/{arxiv_id}"))
    urls.append(("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"))
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
    arxiv_id: str, get_tavily: Callable[[], TavilyTools], fixture_dir: Path = FIXTURE_DIR
) -> PaperText:
    """Fixture zuerst; `get_tavily` wird nur bei Fixture-Miss aufgerufen (spart Credits)."""
    fixture = fixture_dir / f"{arxiv_id}.json"
    if fixture.exists():
        return PaperText(**json.loads(fixture.read_text()))
    tavily = get_tavily()

    for kind, url in _source_urls(arxiv_id):
        result = tavily.read_claims(url)
        for item in result.get("results", []):
            text = item.get("raw_content") or ""
            if len(text) > 2000:  # leere/Fehlerseiten aussortieren
                paper = PaperText(
                    arxiv_id=arxiv_id, url=url, source_kind=kind,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    arxiv_version=latest_arxiv_version(arxiv_id), text=text,
                )
                fixture_dir.mkdir(parents=True, exist_ok=True)
                fixture.write_text(json.dumps(asdict(paper), ensure_ascii=False, indent=1))
                return paper
    raise ExtractionInvalid(f"Kein Volltext fuer {arxiv_id} ueber {[k for k, _ in _source_urls(arxiv_id)]}")


def _norm(s: str) -> str:
    """Whitespace glaetten. PDF-Extraktion liefert Zahlen als '81 .5' -> '81.5'."""
    s = re.sub(r"\s+", " ", s.replace(chr(0xA0), " ")).strip().lower()
    return re.sub(r"(\d) \.(\d)", r"\1.\2", s)


def _windows(text: str, keyword: str, width: int = 2500, max_chars: int = 14000) -> str:
    """Nur Textstellen um das Stichwort an das Modell geben (Tabellen), nicht das ganze Paper."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(re.escape(keyword), text, flags=re.I):
        lo, hi = max(0, m.start() - width), min(len(text), m.end() + width)
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
    return "\n...\n".join(text[a:b] for a, b in spans)[:max_chars]


def verify_extraction(data: dict, paper_text: str) -> float:
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
    return value / 100 if value > 1 else value  # Prozentangabe -> Bruch


def extract_target(
    llm: LLMClient, paper: PaperText, method: str, dataset: str, metric: str
) -> TargetEvidence:
    context = _windows(paper.text, dataset)
    if not context:
        raise ExtractionInvalid(f"'{dataset}' kommt im Quelltext nicht vor")
    resp = llm.complete("fast", [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": (
            f"Find the reported {metric} of method '{method}' on dataset '{dataset}'.\n"
            'JSON keys: "value" (number exactly as printed, e.g. 81.5), "table" (e.g. "Table 2"), '
            '"quote" (a VERBATIM excerpt copied from the text that contains the number, ideally the '
            "table row). Do not compute or guess; if not printed, use null for value.\n\n"
            f"TEXT:\n{context}"
        )},
    ], temperature=0.0)
    m = re.search(r"\{.*\}", resp.content, flags=re.S)
    if not m:
        raise ExtractionInvalid("Modellantwort enthaelt kein JSON")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ExtractionInvalid(f"JSON ungueltig: {exc}") from exc
    value = verify_extraction(data, paper.text)
    return TargetEvidence(
        value=value, quote=str(data["quote"]), table=data.get("table"), url=paper.url,
        source_kind=paper.source_kind, retrieved_at=paper.retrieved_at, sha256=paper.sha256,
        arxiv_version=paper.arxiv_version,
    )

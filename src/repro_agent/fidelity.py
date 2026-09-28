"""Methodentreue: setzt der Agenten-Code die Methode aus dem Paper um, oder nur eine Zahl?

Der Harness prueft Vorhersagen, nicht die Methode. Hier zwei Schritte:

1. Bausteinliste pro Paper (`benchmark/components/<paper_id>.json`): tragende Bausteine der Methode
   mit woertlichem Zitat aus dem Paper und optional einem festen Wert (z. B. Kernel 25). Wird EINMAL
   extrahiert, von Hand geprueft und eingefroren, damit alle Generierungen und Arme gegen dieselbe
   Liste geprueft werden.
2. Pruefer: ordnet jedem Baustein eine Stelle im Agenten-Code zu und zitiert sie woertlich. Ein
   Programm prueft, dass das Zitat im Code steht und (falls vorgegeben) der Wert darin vorkommt.
   "nicht gefunden" ist eine gleichwertige Antwort, keine Notloesung.

Die Pruefung belegt, dass Code EXISTIERT, nicht dass er BENUTZT wird (dafuer: Ausfuehrungsabdeckung,
Identitaetstest, siehe CLAUDE.md).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .claims_source import PaperText, _norm, _windows
from .llm import LLMClient

COMPONENT_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "components"
STATUSES = ("umgesetzt", "nicht umgesetzt", "nicht gefunden")


@dataclass
class Component:
    id: str
    name: str
    description: str
    quote: str                    # woertlich aus dem Paper
    critical: bool = True         # tragender Baustein der Methode
    value: str | None = None      # fester Wert, der im Code-Zitat vorkommen muss (z. B. "25")
    reviewed: bool = False        # von Hand geprueft und eingefroren


@dataclass
class ComponentResult:
    id: str
    status: str                   # umgesetzt | teilweise | nicht umgesetzt | nicht gefunden
    code_quote: str | None
    note: str


@dataclass
class FidelityReport:
    overall: str                  # ja | teilweise | nein
    results: list[ComponentResult] = field(default_factory=list)


# ---------------------------------------------------------------- Hilfen

def _last_json(content: str, opener: str):
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(re.escape(opener), content))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        return value
    return None


def load_components(paper_id: str, directory: Path = COMPONENT_DIR) -> list[Component]:
    path = directory / f"{paper_id}.json"
    return [Component(**c) for c in json.loads(path.read_text())["components"]]


def save_components(paper_id: str, components: list[Component], meta: dict, directory: Path = COMPONENT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{paper_id}.json"
    path.write_text(json.dumps({"meta": meta, "components": [asdict(c) for c in components]},
                               ensure_ascii=False, indent=1))
    return path


# ---------------------------------------------------------------- Schritt 1

def extract_components(
    llm: LLMClient, paper: PaperText, method: str, keywords: list[str]
) -> tuple[list[Component], list[str]]:
    """Bausteine der Methode aus dem Paper-Text. Zitat muss im Text stehen, sonst verworfen.
    Zweiter Rueckgabewert: verworfene Zitate (zur Kontrolle)."""
    context = "\n...\n".join(dict.fromkeys(_windows(paper.text, k, width=1800, max_chars=7000) for k in keywords))
    resp = llm.complete("reasoning", [
        {"role": "system", "content": "You analyse how a machine-learning method is defined. Keep reasoning brief."},
        {"role": "user", "content": (
            f"Method: {method}.\nFrom the paper excerpts below, list the components that DEFINE this "
            "method, i.e. the parts an implementation must contain to be this method and not some "
            "other model (architecture parts, preprocessing that is part of the method, fixed "
            "hyperparameters the paper states as part of it). Do not list training tricks of the "
            "experiments or things the paper only mentions for other methods.\n"
            'Answer with a JSON list of {"id": short_snake_case, "name": ..., "description": one '
            'sentence what the code must do, "quote": a VERBATIM excerpt (max 300 chars) from the '
            'excerpts stating it, "critical": true if the method is not this method without it, '
            '"value": the exact number/setting the paper fixes for it (as a string) or null}. '
            "End with the JSON list.\n\nEXCERPTS:\n" + context[:24000]
        )},
    ], temperature=0.0, max_tokens=8000, timeout_s=600)
    raw = _last_json(resp.content, "[") or []
    haystack = _norm(paper.text)
    kept, dropped = [], []
    for item in raw:
        if not isinstance(item, dict) or not all(item.get(k) for k in ("id", "name", "description", "quote")):
            dropped.append(f"unvollstaendig: {str(item)[:80]}")
            continue
        if _norm(str(item["quote"])) not in haystack:
            dropped.append(f"Zitat nicht im Text: {str(item['quote'])[:110]}")
            continue
        value = item.get("value")
        kept.append(Component(
            id=str(item["id"]), name=str(item["name"]), description=str(item["description"]),
            quote=str(item["quote"]), critical=bool(item.get("critical", True)),
            value=None if value in (None, "", "null") else str(value),
        ))
    return kept, dropped


# ---------------------------------------------------------------- Schritt 2

def numbered(code: str) -> str:
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(code.splitlines(), 1))


def check_fidelity(llm: LLMClient, components: list[Component], code: str) -> FidelityReport:
    listing = "\n".join(
        f'- id={c.id}: {c.name}. {c.description}' + (f" Required value: {c.value}." if c.value else "")
        for c in components
    )
    resp = llm.complete("reasoning", [
        {"role": "system", "content": (
            "You review whether a Python training script implements a specific method. Be strict: a "
            "component counts as implemented only if the code really does it. 'not_found' and "
            "'not_implemented' are normal, correct answers; do not stretch a similar piece of code "
            "to fit a component. Keep reasoning brief."
        )},
        {"role": "user", "content": (
            f"Components the method requires:\n{listing}\n\nScript (line-numbered):\n{numbered(code)}\n\n"
            'Answer with a JSON list, one object per component id: {"id": ..., "status": one of '
            '"implemented" | "not_implemented" | "not_found", "code_quote": the VERBATIM code lines '
            "(without the line-number prefix) that implement it, or null, \"note\": one short sentence}. "
            "End with the JSON list."
        )},
    ], temperature=0.0, max_tokens=8000, timeout_s=600)
    raw = _last_json(resp.content, "[") or []
    by_id = {str(r.get("id")): r for r in raw if isinstance(r, dict)}
    code_norm = _norm(code)
    results = []
    for c in components:
        r = by_id.get(c.id)
        if r is None:
            results.append(ComponentResult(c.id, "nicht gefunden", None, "Pruefer lieferte keine Antwort zu diesem Baustein"))
            continue
        status = {"implemented": "umgesetzt", "not_implemented": "nicht umgesetzt",
                  "not_found": "nicht gefunden"}.get(str(r.get("status")), "nicht gefunden")
        quote = r.get("code_quote")
        note = str(r.get("note", ""))
        if status == "umgesetzt":
            if not quote or _norm(str(quote)) not in code_norm:
                status, note = "nicht gefunden", "Code-Zitat steht nicht im Code (verworfen): " + note
            elif c.value and c.value not in str(quote):
                status, note = "teilweise", f"Wert {c.value} steht nicht im Zitat: " + note
        results.append(ComponentResult(c.id, status, str(quote) if quote else None, note))
    return FidelityReport(overall=aggregate(components, results), results=results)


def aggregate(components: list[Component], results: list[ComponentResult]) -> str:
    """ja: alle tragenden Bausteine umgesetzt; nein: keiner; sonst teilweise."""
    status = {r.id: r.status for r in results}
    critical = [c for c in components if c.critical]
    if not critical:
        return "nicht pruefbar"
    ok = sum(status.get(c.id) == "umgesetzt" for c in critical)
    if ok == len(critical):
        return "ja"
    if ok == 0:
        return "nein"
    return "teilweise"

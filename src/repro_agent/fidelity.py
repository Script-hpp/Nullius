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
from .harness.coverage_check import Coverage, quote_coverage
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


CHECKER_SYSTEM = (
    "You review whether a Python training script implements a specific method. Be strict: a "
    "component counts as implemented only if the code really does it. 'not_found' and "
    "'not_implemented' are normal, correct answers; do not stretch a similar piece of code "
    "to fit a component. Keep reasoning brief."
)

# Nur typografische Zeichen vereinheitlichen; Bezeichner und Zahlen bleiben exakt (case-sensitiv).
_TYPO = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2212": "-", "\u00a0": " ",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
})
_LINE_PREFIX = re.compile(r"^\s*\d+\|\s?", flags=re.M)   # unser Zeilennummern-Format, falls mitzitiert


def _code_norm(s: str, strip_prefix: bool = False) -> str:
    """Leerraum und Zeilenumbrueche vereinheitlichen (auch um Operatoren: `k=5` == `k = 5`)."""
    if strip_prefix:
        s = _LINE_PREFIX.sub("", s)
    return re.sub(r"\s+", "", s.translate(_TYPO))


def value_in_quote(value: str, quote: str) -> bool:
    """Wert als ganzes Token: `5` steht nicht in `kernel_size=25`, `0.5` nicht in `10.55`."""
    return re.search(rf"(?<![\w.]){re.escape(value)}(?![\w]|\.\d)", quote) is not None


def default_overrides(code: str, value: str, quote: str) -> list[str]:
    """Standardargumente `param=VALUE`, deren Wert beim Aufruf mit einer Konstante ueberschrieben wird
    (`Model(a, b, 5)` oder `Model(a, b, kernel_size=5)`). Nur Parameter, die im zitierten Code vorkommen.
    Nicht-konstante Argumente (Variablen) sind nicht entscheidbar und zaehlen nicht als Ueberschreibung."""
    import ast

    try:
        tree = ast.parse(code)
        want = float(value)
    except (SyntaxError, ValueError):
        return []

    def const_num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        return None

    targets = []   # (aufgerufener Name, Parametername, Position ohne self)
    for node in ast.walk(tree):
        fns = []
        if isinstance(node, ast.ClassDef):
            fns = [(node.name, f) for f in node.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"]
        elif isinstance(node, ast.FunctionDef) and node.name != "__init__":
            fns = [(node.name, node)]
        for callee, fn in fns:
            params = [a.arg for a in fn.args.args if a.arg != "self"]
            defaults = fn.args.defaults
            for k, d in enumerate(defaults):
                name = fn.args.args[len(fn.args.args) - len(defaults) + k].arg
                if name != "self" and const_num(d) == want and name in quote:
                    targets.append((callee, name, params.index(name)))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        for callee, name, pos in targets:
            if fname != callee:
                continue
            for kw in node.keywords:
                if kw.arg == name and const_num(kw.value) not in (None, want):
                    found.append(f"{callee}({name}={const_num(kw.value):g}) ueberschreibt den Standardwert {value} (Zeile {node.lineno})")
            if len(node.args) > pos and const_num(node.args[pos]) not in (None, want):
                found.append(f"{callee}(..., {const_num(node.args[pos]):g}) ueberschreibt {name}={value} (Zeile {node.lineno})")
    return found


def _ask_checker(llm: LLMClient, listing: str, code: str) -> tuple[list | None, str]:
    """(Ergebnisliste, Ausfallgrund). Abgeschnitten oder unlesbar: einmal neu, mit doppelter Grenze."""
    reason = ""
    for max_tokens, temperature in ((8000, 0.0), (16000, 0.2)):   # Temperatur anders: sonst kaeme Cache-Treffer
        resp = llm.complete("extract", [
            {"role": "system", "content": CHECKER_SYSTEM},
            {"role": "user", "content": (
                f"Components the method requires:\n{listing}\n\nScript (line-numbered):\n{numbered(code)}\n\n"
                'Answer with a JSON list, one object per component id: {"id": ..., "status": one of '
                '"implemented" | "not_implemented" | "not_found", "code_quote": the VERBATIM code lines '
                "(without the line-number prefix) that implement it, or null, \"note\": one short sentence}. "
                "If a component lists a required value, code_quote must include the line where that value is set "
                "(assignment, default argument or call argument), not only the line that uses a variable. "
                "End with the JSON list."
            )},
        ], temperature=temperature, max_tokens=max_tokens, timeout_s=600)
        raw = _last_json(resp.content, "[")
        if isinstance(raw, list) and any(isinstance(r, dict) and "id" in r for r in raw):
            return raw, ""
        reason = ("Ausgabe abgeschnitten (finish_reason length)" if resp.finish_reason == "length"
                  else "keine lesbare JSON-Liste")
    return None, reason


def check_fidelity(llm: LLMClient, components: list[Component], code: str) -> FidelityReport:
    """Ein technischer Ausfall (unlesbare Antwort, fehlender Baustein) ist nie ein Urteil: Ergebnis
    'nicht geprueft', nicht 'nicht gefunden' oder 'nein'."""
    listing = "\n".join(
        f'- id={c.id}: {c.name}. {c.description}' + (f" Required value: {c.value}." if c.value else "")
        for c in components
    )
    raw, reason = _ask_checker(llm, listing, code)
    if raw is None:
        note = f"technischer Ausfall des Pruefers: {reason}"
        return FidelityReport("nicht geprueft", [ComponentResult(c.id, "nicht geprueft", None, note) for c in components])
    by_id = {str(r.get("id")): r for r in raw if isinstance(r, dict)}
    code_norm = _code_norm(code)
    results = []
    for c in components:
        r = by_id.get(c.id)
        status = {"implemented": "umgesetzt", "not_implemented": "nicht umgesetzt",
                  "not_found": "nicht gefunden"}.get(str(r.get("status"))) if r else None
        if status is None:
            results.append(ComponentResult(c.id, "nicht geprueft", None, "technischer Ausfall: Antwort fehlt oder Status unlesbar"))
            continue
        quote = r.get("code_quote")
        note = str(r.get("note", ""))
        if status == "umgesetzt":
            if not quote or _code_norm(str(quote), strip_prefix=True) not in code_norm:
                status, note = "nicht gefunden", "Code-Zitat steht nicht im Code (verworfen): " + note
            elif c.value and not value_in_quote(c.value, str(quote)):
                status, note = "teilweise", f"Wert {c.value} steht nicht im Zitat: " + note
            elif c.value and (over := default_overrides(code, c.value, str(quote))):
                status, note = "teilweise", "Standardwert wird ueberschrieben: " + "; ".join(over)
        results.append(ComponentResult(c.id, status, str(quote) if quote else None, note))
    return FidelityReport(overall=aggregate(components, results), results=results)


def apply_coverage(
    components: list[Component], report: FidelityReport, code: str, cov: Coverage | None
) -> tuple[FidelityReport, list[str]]:
    """Kombinationsregel (calibration_fidelity_v2.md): Abdeckung darf nur HERABSTUFEN.
    Nur bei Pruefer-Urteil `ja`: umgesetzter Baustein, dessen zitierte Zeilen nie liefen, gilt als nicht umgesetzt
    (Gesamturteil dann `teilweise` oder `nein` ueber `aggregate`). `teilweise`/`nein` bleiben unveraendert.
    Ohne Abdeckung bleibt das Urteil, mit Hinweis."""
    if cov is None:
        return report, ["Ausfuehrungsabdeckung nicht verfuegbar: Pruefer-Urteil unveraendert (Existenzpruefung)."]
    if report.overall != "ja":
        return report, []
    notes, results = [], []
    for r in report.results:
        if r.status == "umgesetzt" and r.code_quote:
            cs = quote_coverage(code, r.code_quote, cov)
            if cs == "nicht ausgefuehrt":
                notes.append(f"{r.id}: zitierte Zeilen wurden im Lauf nie ausgefuehrt")
                r = ComponentResult(r.id, "nicht umgesetzt", r.code_quote, "Zitat nie ausgefuehrt (Abdeckung): " + r.note)
        results.append(r)
    return FidelityReport(aggregate(components, results), results), notes


def aggregate(components: list[Component], results: list[ComponentResult]) -> str:
    """ja: alle tragenden Bausteine umgesetzt; nein: keiner; sonst teilweise.
    Fehlt bei einem tragenden Baustein ein Ergebnis (technischer Ausfall): nicht geprueft."""
    status = {r.id: r.status for r in results}
    critical = [c for c in components if c.critical]
    if not critical:
        return "nicht pruefbar"
    if any(status.get(c.id, "nicht geprueft") == "nicht geprueft" for c in critical):
        return "nicht geprueft"
    ok = sum(status.get(c.id) == "umgesetzt" for c in critical)
    if ok == len(critical):
        return "ja"
    if ok == 0:
        return "nein"
    return "teilweise"

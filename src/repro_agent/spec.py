"""Steckbrief (S1, docs/recall_steckbrief_plan.md): wenige entscheidende Angaben mit Beleg statt eines Zitatbergs.

Die KI liest, ein Programm prueft (wie Rolle 2/4): Ein LLM-Aufruf fuellt feste Schluessel je Task-Familie aus den schon
belegten Setup-Details und einem kleinen Repo-Ausschnitt. Jeder Eintrag braucht ein woertliches Zitat aus einem gelesenen
Dokument, Zahlen des Werts muessen im Zitat stehen. Widersprueche loest eine feste Rangregel (Startskript > Config > Paper >
Code > README > argparse-Default), bevor ein Modell sie sieht. Der Modellname des Claims schlaegt jede Modellwahl im Skript.

Die Schluessel nennen nur Themen, nie Werte oder Methodenwissen (gegen Ueberanpassung an die Dev-Einheiten).
Der Steckbrief enthaelt Werte, Formeln und kurze Beschreibungen, keine ganzen Autorendateien (Modus bleibt Nachbau).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .claims_source import _norm
from .setup_details import (
    CALL_RE, OPTIONAL_RE, RANKS, START_RE, Doc, SetupDetail, family_for, function_body, numbers_in_quote, quote_in_text,
    source_rank,
)

# ---- Schluessel je Familie ------------------------------------------------------------------------------------------
# (Schluessel, Beschreibung fuer den Extraktor). Zahlenschluessel werden im Code erzwungen (S2), Textschluessel einzeln geprueft (S3).
NUMERIC_COMMON = [
    ("lr", "learning rate (one number)"),
    ("weight_decay", "weight decay / L2 coefficient (one number)"),
    ("epochs", "maximum number of training epochs (one integer)"),
    ("batch_size", "batch size (one integer)"),
    ("dropout", "dropout probability (one number; if several dropouts differ, give a text value instead)"),
    ("hidden", "hidden size of the model (one integer; if it depends on heads or layers, give a text value instead)"),
    ("layers", "number of layers of the model's main block (one integer)"),
    ("patience", "early-stopping patience in epochs (one integer)"),
]
NUMERIC_FAMILY = {
    "graph": [("heads", "number of attention heads, if the model has attention (one integer)")],
    "forecast": [("seq_len", "input window / look-back length (one integer)")],
}
TEXT_COMMON = [
    ("optimizer", "optimizer name"),
    ("lr_schedule", "learning-rate schedule as a formula or rule (e.g. lr * factor ** epoch), or 'constant'"),
    ("early_stopping", "what early stopping monitors and which weights are restored for testing"),
    ("activation", "activation function(s) of the model"),
    ("init", "weight initialisation"),
    ("model_core", "the core computation of the model in 1 to 3 sentences (what the forward pass does)"),
]
TEXT_FAMILY = {
    "graph": [
        ("feature_norm", "how node features are normalised before training (e.g. row-normalised to sum 1, none)"),
        ("adj_norm", "how the adjacency / propagation matrix is normalised"),
        ("self_loops", "whether self-loops are added"),
    ],
    "forecast": [
        ("input_norm", "normalisation of each input window inside the model (e.g. subtract last value, RevIN), or none"),
        ("channel_mode", "whether channels are modelled individually (shared or separate weights) or jointly"),
    ],
}
INT_KEYS = {"epochs", "batch_size", "hidden", "layers", "patience", "heads", "seq_len"}
EXCERPT_CHARS = 15_000      # Repo-Ausschnitt fuer den Steckbrief-Aufruf (nie im Code-Prompt)
PIECE_CHARS = 6_000
QUOTE_SHOW = 160            # Zitatlaenge im Prompt
NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
MODEL_SEL_RE = re.compile(r"\bmodel(?:_name|_type)?\b\s*(?:=|:|\s)\s*['\"]?([A-Za-z][\w-]*)", re.I)
BINARY_EXT = re.compile(r"\.(pkl|pickle|npz|npy|pt|pth|bin|h5|hdf5|zip|gz|tar|png|jpe?g|gif|pdf|so|pyc|ckpt|mat|csv|tsv|parquet)$", re.I)
DATA_DIR = re.compile(r"(^|/)(data|dataset|datasets|checkpoints?|logs?|results?|figures?|pics?|img|images)/", re.I)


def keys_for(family: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    return NUMERIC_COMMON + NUMERIC_FAMILY.get(family, []), TEXT_COMMON + TEXT_FAMILY.get(family, [])


def model_name_of(claim_text: str) -> str:
    """Modellname des Claims: Text vor ' erreicht ' (papers.yaml), ohne Variante nach '/' (PatchTST/42 -> PatchTST)."""
    head = claim_text.split(" erreicht ")[0].strip() if " erreicht " in claim_text else claim_text.split()[0]
    return head.split("/")[0].strip()


# ---- Quelltext-Filter (S4) ------------------------------------------------------------------------------------------
def is_source_text(path: str, text: str) -> bool:
    """Nur lesbarer Quelltext/Konfiguration: keine Binaer- oder Datendateien (SGC: sechs Pickles, ~153 000 Zeichen)."""
    if BINARY_EXT.search(path) or DATA_DIR.search(path):
        return False
    if "\x00" in text:
        return False
    sample = text[:20_000]
    return not sample or sample.count("\ufffd") / len(sample) < 0.01


# ---- Repo-Ausschnitt ------------------------------------------------------------------------------------------------
def _path(doc: Doc) -> str:
    return doc.name.split(":", 1)[-1].split("@")[0]


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def model_files(docs: list[Doc], model: str) -> list[Doc]:
    """Repo-Dateien, die die Klasse des Claim-Modells definieren oder so heissen (models/NLinear.py, models/dgi.py)."""
    m = model.lower()
    cls = re.compile(rf"^\s*class\s+{re.escape(model)}\b", re.M | re.I)
    out = [d for d in docs if d.name.startswith("repo:") and _path(d).endswith(".py") and (_stem(_path(d)) == m or cls.search(d.text))]
    return sorted(out, key=lambda d: (_stem(_path(d)) != m, len(d.text)))


def _imported_docs(src: list[Doc], docs: list[Doc]) -> list[Doc]:
    from .repo_source import IMPORT_RE
    by_mod = {}
    for d in docs:
        if d.name.startswith("repo:") and _path(d).endswith(".py"):
            by_mod.setdefault(_stem(_path(d)), d)
    out = []
    for d in src:
        for a, names, b in IMPORT_RE.findall(d.text):
            mods = [(a or b).strip(".").rsplit(".", 1)[-1].lower()] + [n.strip().lower() for n in names.split(",") if n.strip()]
            for mod in mods:
                hit = by_mod.get(mod)
                if hit is not None and hit not in src and hit not in out:
                    out.append(hit)
    return out


def _called_bodies(details: list[SetupDetail], docs: list[Doc]) -> list[tuple[str, str]]:
    """Ruempfe der in den Angaben aufgerufenen Funktionen (preprocess_features, adjust_learning_rate ...), aus den gelesenen Dateien."""
    py = [d for d in docs if d.name.startswith("repo:") and _path(d).endswith(".py")]
    out, seen = [], set()
    for det in details:
        if getattr(det, "excluded", ""):
            continue
        for call in CALL_RE.findall(f"{det.value} {det.quote}"):
            name = call.rsplit(".", 1)[-1]
            if name in seen or len(name) < 4:
                continue
            for d in py:
                body = function_body(d.text, name)
                if body:
                    seen.add(name)
                    out.append((d.name, body))
                    break
    return out


def focused_excerpt(docs: list[Doc], model: str, dataset: str, details: list[SetupDetail], budget: int = EXCERPT_CHARS
                    ) -> tuple[str, list[str]]:
    """Kleiner Repo-Ausschnitt: Modelldatei des Claims, deren importierte Layer, aufgerufene Vorverarbeitungs-/Schedule-Funktionen,
    Startskripte fuer Datensatz oder Modell. Nur Quelltext. Rueckgabe (Text, Namen der Stuecke)."""
    repo = [d for d in docs if d.name.startswith("repo:") and is_source_text(_path(d), d.text)]
    pieces: list[tuple[str, str]] = []
    mf = model_files(repo, model)[:2]
    pieces += [(d.name, d.text) for d in mf]
    pieces += [(d.name, d.text) for d in _imported_docs(mf, repo)[:3]]
    pieces += [(f"{name} (function body)", body) for name, body in _called_bodies(details, repo)[:4]]
    ds, m = (re.split(r"[^a-z0-9]+", dataset.lower()) + [""])[0], model.lower()
    starts = [d for d in repo if START_RE.search(_path(d)) and (ds and ds in d.text.lower() or m in d.text.lower())]
    starts.sort(key=lambda d: (not _path(d).endswith(".sh"), m not in d.text.lower(), len(d.text)))
    pieces += [(d.name, d.text) for d in starts[:2]]
    out, names, used, seen = [], [], 0, set()
    for name, text in pieces:
        if name in seen:
            continue
        seen.add(name)
        chunk = text[:PIECE_CHARS]
        if used + len(chunk) > budget:
            chunk = chunk[: max(0, budget - used)]
        if len(chunk) < 200:
            break
        out.append(f"=== {name} ===\n{chunk}")
        names.append(name)
        used += len(chunk)
    return "\n\n".join(out), names


# ---- Steckbrief -----------------------------------------------------------------------------------------------------
@dataclass
class SpecItem:
    key: str
    value: str
    quote: str
    source: str
    rank: str
    number: float | int | None = None   # nur Zahlenschluessel mit eindeutigem Zahlwert
    enforce: bool = False               # S2: im CONFIG-Block erzwingen
    alternatives: list[dict] = field(default_factory=list)   # unterlegene Werte anderer Quellen (Dossier, nie im Prompt)
    note: str = ""


@dataclass
class Spec:
    model: str
    family: str
    items: list[SpecItem] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    excerpt_docs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def get(self, key: str) -> SpecItem | None:
        return next((i for i in self.items if i.key == key), None)

    def enforced(self) -> list[SpecItem]:
        return [i for i in self.items if i.enforce]

    def checkable(self) -> list[SpecItem]:
        """S3: alles, was nicht per CONFIG erzwungen wird."""
        return [i for i in self.items if not i.enforce]


SPEC_SYSTEM = "You fill a reproduction spec from verified sources. Keep reasoning brief."


def build_prompt(details: list[SetupDetail], excerpt: str, claim_text: str, dataset: str, model: str, family: str) -> str:
    num, txt = keys_for(family)
    keys = "\n".join(f"- {k}: {d}" for k, d in num + txt)
    lines = []
    for d in details:
        if getattr(d, "excluded", ""):
            continue
        rank = getattr(d, "source_rank", "") or source_rank(d.source, d.quote)
        lines.append(f"- {d.topic}: {d.value} ({d.source} [{rank}]: \"{d.quote}\")")
    return (
        f"Goal: reproduce this result: {claim_text} (dataset: {dataset}). The model to build is {model}.\n"
        "Fill the spec keys below for exactly this model and dataset. Use only the verified setup details and the code excerpt "
        "below. For each value give a VERBATIM quote (max 300 chars) copied from the document it comes from, and that "
        "document's exact name as source: \"paper\", \"repo-readme\" or the name after === in the excerpt (without the "
        "'(function body)' suffix). If several sources give a value for the same key, report each as its own entry. "
        "Skip values that belong to another model, another dataset or that a source only gives as optional advice. Skip a "
        "key if no source states it; never guess. For number keys give the bare number as written (e.g. 0.005, 336).\n"
        f"Keys:\n{keys}\n\n"
        "Answer with a JSON list of {\"key\": one of the keys, \"value\": the value, \"quote\": verbatim quote, \"source\": "
        "document name} and end with the JSON list.\n\n"
        "VERIFIED SETUP DETAILS:\n" + ("\n".join(lines) or "(none)") + "\n\n"
        "CODE EXCERPT:\n" + (excerpt or "(none)")
    )


def _parse_list(content: str) -> list:
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\[", content or ""))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def parse_number(key: str, value: str) -> float | int | None:
    v = value.strip().rstrip(".,;").replace("_", "")
    if not NUM_RE.match(v):
        return None
    x = float(v)
    if key in INT_KEYS:
        return int(x) if x == int(x) else None
    return int(x) if x == int(x) and "." not in v and "e" not in v.lower() else x


def _number_in_quote(value: str, quote: str) -> bool:
    """Wie numbers_in_quote, dazu wissenschaftliche Schreibweise (5e-3 belegt durch '5e-3' oder '0.005')."""
    if numbers_in_quote(value, quote):
        return True
    try:
        x = float(value)
    except ValueError:
        return False
    for tok in re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", quote):
        try:
            if float(tok) == x:
                return True
        except ValueError:
            continue
    return False


def other_model_selected(quote: str, model: str) -> str:
    """Modellname-Regel: setzt das Zitat eine Modellwahl auf ein anderes Modell (model_name=DLinear bei NLinear), gilt es nicht."""
    for m in MODEL_SEL_RE.finditer(quote):
        name = m.group(1)
        if name.lower() != model.lower() and not name.lower().startswith(("str", "int", "float", "none", "args", "self")):
            return name
    return ""


def verify_entries(raw: list, docs: list[Doc], model: str, family: str) -> tuple[list[SpecItem], list[dict]]:
    """Deterministisch: bekannter Schluessel, bekannte Quelle, Zitat im Dokument, Zahl im Zitat, keine fremde Modellwahl."""
    num, txt = keys_for(family)
    num_keys, txt_keys = {k for k, _ in num}, {k for k, _ in txt}
    hay = {d.name: _norm(d.text) for d in docs}
    kept, dropped = [], []
    for it in raw:
        if not isinstance(it, dict):
            dropped.append({"item": it, "reason": "kein Objekt"})
            continue
        key, value, quote, source = (str(it.get(k, "")).strip() for k in ("key", "value", "quote", "source"))
        source = re.sub(r"\s*\(function body\)\s*$", "", source)
        reason = ""
        if key not in num_keys | txt_keys:
            reason = "unbekannter Schluessel"
        elif not (value and quote and source):
            reason = "Feld fehlt"
        elif source not in hay:
            reason = "Quelle unbekannt"
        elif not quote_in_text(quote, hay[source]):
            reason = "Zitat nicht im Quelltext"
        elif not _number_in_quote(value, quote) and re.search(r"\d", value):
            reason = "Zahl des Werts nicht im Zitat"
        elif other_model_selected(quote, model):
            reason = f"andere Modellwahl ({other_model_selected(quote, model)})"
        elif source == "repo-readme" and OPTIONAL_RE.search(quote):
            reason = "optionaler README-Hinweis"
        if reason:
            dropped.append({"item": it, "reason": reason})
            continue
        number = parse_number(key, value) if key in num_keys else None
        kept.append(SpecItem(key, value, quote, source, source_rank(source, quote), number))
    return kept, dropped


def _rank_index(rank: str) -> int:
    return RANKS.index(rank) if rank in RANKS else len(RANKS)


def _same_value(a: SpecItem, b: SpecItem) -> bool:
    if a.number is not None and b.number is not None:
        return float(a.number) == float(b.number)
    return _norm(a.value) == _norm(b.value)


def resolve(items: list[SpecItem], family: str) -> tuple[list[SpecItem], list[str]]:
    """Ein Eintrag je Schluessel: hoechster Rang gewinnt; argparse-Defaults fallen weg, sobald eine hoehere Quelle den Schluessel
    setzt (D-Klasse). Widersprechen sich zwei Quellen GLEICHEN Rangs, wird nichts erzwungen (Hinweis statt Raten).
    Rueckgabe (Eintraege in Schluesselreihenfolge, Schluessel ohne Beleg)."""
    num, txt = keys_for(family)
    order = [k for k, _ in num + txt]
    out, missing = [], []
    for key in order:
        group = sorted((i for i in items if i.key == key), key=lambda i: _rank_index(i.rank))
        if not group:
            missing.append(key)
            continue
        win = group[0]
        alts = [i for i in group[1:] if not _same_value(i, win)]
        win.alternatives = [{"value": a.value, "source": a.source, "rank": a.rank, "quote": a.quote} for a in alts]
        tie = [a for a in alts if a.rank == win.rank]
        if tie:
            win.note = "gleichrangige Quellen widersprechen sich: " + "; ".join(f"{a.value} ({a.source})" for a in tie)
        win.enforce = win.number is not None and not tie
        out.append(win)
    return out, missing


def build_spec(ask, details: list[SetupDetail], docs: list[Doc], claim_text: str, dataset: str) -> Spec:
    """ask(system, user) -> Antworttext. Ein LLM-Aufruf; alles danach ist deterministisch."""
    model, family = model_name_of(claim_text), family_for(dataset)
    excerpt, names = focused_excerpt(docs, model, dataset, details)
    raw = _parse_list(ask(SPEC_SYSTEM, build_prompt(details, excerpt, claim_text, dataset, model, family)))
    kept, dropped = verify_entries(raw, docs, model, family)
    items, missing = resolve(kept, family)
    return Spec(model=model, family=family, items=items, dropped=dropped, excerpt_docs=names, missing=missing)


def _short(q: str) -> str:
    q = " ".join(q.split())
    return q if len(q) <= QUOTE_SHOW else q[: QUOTE_SHOW - 3] + "..."


def _fmt_number(x: float | int) -> str:
    return repr(x)


def format_spec(spec: Spec) -> str:
    """Steckbrief fuer Plan- und Code-Prompt: ersetzt die Liste der Setup-Details (die volle Liste bleibt im Dossier)."""
    if not spec.items:
        return ""
    lines = []
    for i in spec.items:
        val = _fmt_number(i.number) if i.enforce else i.value
        lines.append(f"- {i.key} = {val}   [{i.rank}: \"{_short(i.quote)}\"]")
    missing = f"\nNot stated in any source (choose a standard value and say so in a comment): {', '.join(spec.missing)}." if spec.missing else ""
    return (
        f"Reproduction spec for {spec.model}, checked against the paper and the authors' code. Conflicts between sources are "
        "already resolved (start script > config > paper > code > README > argparse default). Follow every entry exactly; "
        f"if a script or default selects a model other than {spec.model}, ignore that selection.\n"
        + "\n".join(lines) + missing
    )


def config_note(spec: Spec) -> str:
    """Pflicht fuer train.py (S2): literaler CONFIG-Block mit den erzwungenen Schluesseln."""
    enf = spec.enforced()
    if not enf:
        return ""
    body = ", ".join(f'"{i.key}": {_fmt_number(i.number)}' for i in enf)
    return (
        "\n\nStart train.py, right after the imports, with exactly this literal dict and read these hyperparameters only from it "
        "(for example CONFIG[\"lr\"]); do not write these numbers as literals anywhere else. You may add further keys.\n"
        f"CONFIG = {{{body}}}"
    )


def as_dict(spec: Spec) -> dict:
    from dataclasses import asdict
    return asdict(spec)

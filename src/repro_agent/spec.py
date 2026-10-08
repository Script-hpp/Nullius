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
    CALL_RE, CONFIG_RE, OPTIONAL_RE, RANKS, START_RE, Doc, SetupDetail, family_for, function_body, numbers_in_quote, quote_in_text,
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
SCRIPT_CHARS = 4_500        # Start-/Trainingsskripte und Konfigurationen
TRAIN_RE = re.compile(r"\.backward\(\)|optimizer\.step\(\)|\.minimize\(|\.fit\(")   # Trainingsschleife (Skript ohne Standardnamen)
CONFIG_STEMS = {"train", "default", "config", "base", "params", "hparams"}   # allgemeine Trainingskonfiguration neben der des Modells
QUOTE_SHOW = 160            # Zitatlaenge im Prompt
NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
MODEL_SEL_RE = re.compile(r"\bmodel(?:_name|_type)?\b\s*(?:=|:|\s)\s*['\"]?([A-Za-z][\w-]*)", re.I)
BINARY_EXT = re.compile(r"\.(pkl|pickle|npz|npy|pt|pth|bin|h5|hdf5|zip|gz|tar|png|jpe?g|gif|pdf|so|pyc|ckpt|mat|csv|tsv|parquet)$", re.I)
DATA_DIR = re.compile(r"(^|/)(data|dataset|datasets|checkpoints?|logs?|results?|figures?|pics?|img|images)/", re.I)


def keys_for(family: str, numeric_only: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """numeric_only (code_grounded): nur Zahlenschluessel; was nicht per CONFIG erzwungen wird, steht roh im Code-Ausschnitt."""
    if numeric_only:
        return NUMERIC_COMMON + NUMERIC_FAMILY.get(family, []), []
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
    from .repo_source import IMPORT_RE_S4 as IMPORT_RE
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


PREP_RE = re.compile(r"norm|preprocess|schedul|adjust_|scale|standardi|split|load_", re.I)   # Vorverarbeitung und lr-Schedule (Namen, keine Werte)


def _called_bodies(details: list[SetupDetail], docs: list[Doc], scripts: list[Doc] | None = None) -> list[tuple[str, str]]:
    """Ruempfe der aufgerufenen Funktionen (preprocess_features, adjust_learning_rate ...), aus den gelesenen Dateien. Aufrufe stammen aus
    den Angaben (Zitat/Wert) und aus Startskripten/Modelldatei, dort nur Namen der Vorverarbeitung oder des Schedules (PREP_RE)."""
    py = [d for d in docs if d.name.startswith("repo:") and _path(d).endswith(".py")]
    out, seen = [], set()
    calls = [c for det in details if not getattr(det, "excluded", "") for c in CALL_RE.findall(f"{det.value} {det.quote}")]
    calls += [c for d in (scripts or []) if _path(d).endswith(".py") for c in CALL_RE.findall(d.text) if PREP_RE.search(c)]
    for call in calls:
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
    ds, m = (re.split(r"[^a-z0-9]+", dataset.lower()) + [""])[0], model.lower()
    starts = [d for d in repo if START_RE.search(_path(d)) and (ds and ds in d.text.lower() or m in d.text.lower())]
    starts.sort(key=lambda d: (not _path(d).endswith(".sh"), m not in d.text.lower(), len(d.text)))
    trainers = sorted((d for d in repo if _path(d).endswith(".py") and TRAIN_RE.search(d.text) and d not in mf and d not in starts),
                      key=lambda d: len(d.text))[:2]
    cfgs = [d for d in repo if CONFIG_RE.search(_path(d)) and (m in _stem(_path(d)) or _stem(_path(d)).split(".")[0] in CONFIG_STEMS)]
    cfgs.sort(key=lambda d: (m not in _path(d).lower(), len(d.text)))
    # Reihenfolge = Vorrang beim Zeichenbudget: Modelldatei, Startskripte, Modell-Konfiguration, importierte Layer, Rumpfe
    pieces += [(d.name, d.text) for d in mf]
    pieces += [(d.name, d.text[:SCRIPT_CHARS]) for d in starts[:2]]
    pieces += [(d.name, d.text[:SCRIPT_CHARS]) for d in cfgs[:4]]
    scripts = mf + starts[:2] + trainers
    first = _called_bodies([], repo, scripts)                          # Namen der Vorverarbeitung/des Schedules in Skripten
    second = _called_bodies([], repo, [Doc(n, b) for n, b in first])   # zweite Stufe: load_citation -> preprocess_citation
    named = _called_bodies(details, repo)                              # Aufrufe aus den belegten Angaben
    bodies = []
    for x in first + second + named:
        if x not in bodies:
            bodies.append(x)
    rank = [(f"{name} (function body)", body) for name, body in bodies[:10]]
    pieces += rank[:len(first)] + [(d.name, d.text) for d in _imported_docs(mf, repo)[:3]] + rank[len(first):]
    out, names, used, seen = [], [], 0, set()
    for name, text in pieces:
        if (name, text) in seen:   # mehrere Rumpfe derselben Datei tragen denselben Namen
            continue
        seen.add((name, text))
        chunk = text[:PIECE_CHARS]
        if used + len(chunk) > budget:
            chunk = chunk[: max(0, budget - used)]
        if budget - used < 200:
            break          # Budget erschoepft
        if len(chunk.strip()) < 20:
            continue       # leere Datei (__init__.py) ueberspringen; kurze Konfigurationen (hidden_size: 64) bleiben
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
    specific: bool = True               # F4: False = allgemeine Konfiguration (train.conf), die eine Modell-Konfiguration ueberschreibt
    origin: str = "model"               # F1/F4: "raw" = deterministisch aus einer Zeile des Ausschnitts gelesen, sonst vom Extraktor
    tie: list[dict] = field(default_factory=list)   # F4: gleichrangige, widersprechende Werte (nicht erzwungen)


@dataclass
class Spec:
    model: str
    family: str
    items: list[SpecItem] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    excerpt_docs: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    excerpt: str = ""                                          # code_grounded: roher Ausschnitt fuer Plan-, Code- und Reparatur-Prompt
    dropped_defaults: list[str] = field(default_factory=list)  # code_grounded: argparse-Flags, deren Default entfernt wurde

    def get(self, key: str) -> SpecItem | None:
        return next((i for i in self.items if i.key == key), None)

    def enforced(self) -> list[SpecItem]:
        return [i for i in self.items if i.enforce]

    def checkable(self) -> list[SpecItem]:
        """S3: alles, was nicht per CONFIG erzwungen wird."""
        return [i for i in self.items if not i.enforce]


SPEC_SYSTEM = "You fill a reproduction spec from verified sources. Keep reasoning brief."


def build_prompt(details: list[SetupDetail], excerpt: str, claim_text: str, dataset: str, model: str, family: str,
                 numeric_only: bool = False) -> str:
    num, txt = keys_for(family, numeric_only)
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


# ---- F1 / F4 (code_grounded): Werte deterministisch aus Zeilen des Ausschnitts lesen ------------------------------------------
# Feste Tabelle Schluesselname im Code -> Schluessel des Steckbriefs (nur Namen, nie Werte). Nur skalare Zahlen; Listen (hid_units = [8]) fallen weg.
KEY_ALIASES = {
    "lr": ("lr", "learning_rate", "learning-rate", "init_lr"),
    "weight_decay": ("weight_decay", "weight-decay", "wd", "l2_coef", "l2_reg", "l2_lambda", "l2"),
    "epochs": ("epochs", "num_epochs", "nb_epochs", "n_epochs", "max_epochs", "max_epoch", "train_epochs"),
    "batch_size": ("batch_size", "batch-size", "bs"),
    "dropout": ("dropout", "dropout_prob", "dropout_rate", "dropout_p"),
    "hidden": ("hidden", "hidden_size", "hidden_dim", "hidden_channels", "hid_dim", "n_hid", "num_hidden"),
    "layers": ("num_layers", "n_layers", "nb_layers", "layers", "e_layers"),
    "patience": ("patience", "early_stopping_patience", "nb_patience"),
    "seq_len": ("seq_len", "input_len", "lookback", "look_back"),
}
ALIAS_KEY = {alias: key for key, names in KEY_ALIASES.items() for alias in names}
_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
ASSIGN_RE = re.compile(rf"^(?:export\s+)?([A-Za-z_]\w*)\s*(?:=|:)\s*(?:!!float\s+)?({_NUM})\s*[,;]?\s*(?:#.*)?$")   # nur Zeilen ohne Einrueckung
FLAG_NUM_RE = re.compile(rf"--([A-Za-z_][\w-]*)(?:\s+|=)({_NUM})(?![\w.])")
NUM_TOKEN_RE = re.compile(r"\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+|\d+(?:[eE][+-]?\d+)?")


def _doc_path(name: str) -> str:
    return name.split(":", 1)[-1].split("@")[0].replace(" (function body)", "")


def _is_specific(source: str, model: str) -> bool:
    """Konfigurationsdatei mit dem Modellnamen im Pfad (gat.conf.yaml) gilt vor der allgemeinen (train.conf.yaml). Andere Quellen: immer."""
    path = _doc_path(source)
    return model.lower() in path.lower() if CONFIG_RE.search(path) else True


def raw_items(docs: list[Doc], names: list[str], model: str, family: str) -> list[SpecItem]:
    """F1 (Startskript) und F4 (c, Konfiguration): Zuweisungen `name = Zahl`, `name: Zahl` und `--flag Zahl` in den Startskripten und
    Konfigurationsdateien des Ausschnitts. Steht ein Schluessel in einer Datei mit mehreren verschiedenen Werten (mehrere Laeufe in einem
    Shell-Skript), wird er dort nicht gelesen. Deterministisch, kein LLM."""
    num_keys = {k for k, _ in keys_for(family, True)[0]}
    want = {n.replace(" (function body)", "") for n in names}
    out: list[SpecItem] = []
    for d in docs:
        path = _doc_path(d.name)
        if d.name not in want or not (START_RE.search(path) or CONFIG_RE.search(path)):
            continue
        found: dict[str, list[tuple[str, str]]] = {}
        for line in d.text.splitlines():
            m = ASSIGN_RE.match(line.rstrip())
            hits = [(m.group(1), m.group(2))] if m else []
            if path.endswith(".sh"):
                hits += FLAG_NUM_RE.findall(line)
            for name, value in hits:
                key = ALIAS_KEY.get(name.lower().replace("-", "_")) or ALIAS_KEY.get(name.lower())
                if key in num_keys:
                    found.setdefault(key, []).append((value, line.strip()))
        for key, occ in found.items():
            if len({float(v) for v, _ in occ}) != 1:
                continue
            value, quote = occ[0]
            number = parse_number(key, value)
            if number is None:
                continue
            out.append(SpecItem(key, value, quote, d.name, source_rank(d.name, quote), number, specific=_is_specific(d.name, model), origin="raw"))
    return out


def paper_numbers(details: list[SetupDetail], items: list[SpecItem]) -> set[float]:
    """Zahlen aus belegten Paper-Zitaten (Rolle 4 und Steckbrief-Eintraege der Quelle 'paper'): Belege, die einen Konfigurationswert stuetzen."""
    texts = [d.quote for d in details if getattr(d, "source", "") == "paper" and not getattr(d, "excluded", "")]
    texts += [i.quote for i in items if i.source == "paper"]
    nums: set[float] = set()
    for t in texts:
        for tok in NUM_TOKEN_RE.findall(t):
            try:
                nums.add(float(tok))
            except ValueError:
                pass
    return nums


def normalize_source(source: str, names: list[str]) -> str:
    """F4 (a): Quelle ohne `@commit` (oder ohne `repo:`) auf die eindeutige Dokumentzeile abbilden. Unbekannt oder mehrdeutig: unveraendert."""
    if source in names:
        return source
    base = source.removeprefix("repo:").split("@")[0]
    hits = [n for n in names if n.startswith("repo:") and n.removeprefix("repo:").split("@")[0] == base]
    return hits[0] if len(hits) == 1 else source


def verify_entries(raw: list, docs: list[Doc], model: str, family: str, numeric_only: bool = False, normalize: bool = False
                   ) -> tuple[list[SpecItem], list[dict]]:
    """Deterministisch: bekannter Schluessel, bekannte Quelle, Zitat im Dokument, Zahl im Zitat, keine fremde Modellwahl.
    normalize (F4 a, code_grounded): fehlt der `@commit` in der Quelle, wird sie auf die eindeutige Dokumentzeile abgebildet."""
    num, txt = keys_for(family, numeric_only)
    num_keys, txt_keys = {k for k, _ in num}, {k for k, _ in txt}
    hay = {d.name: _norm(d.text) for d in docs}
    kept, dropped = [], []
    for it in raw:
        if not isinstance(it, dict):
            dropped.append({"item": it, "reason": "kein Objekt"})
            continue
        key, value, quote, source = (str(it.get(k, "")).strip() for k in ("key", "value", "quote", "source"))
        source = re.sub(r"\s*\(function body\)\s*$", "", source)
        if normalize:
            source = normalize_source(source, list(hay))
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
        kept.append(SpecItem(key, value, quote, source, source_rank(source, quote), number, specific=_is_specific(source, model)))
    return kept, dropped


def _rank_index(rank: str) -> int:
    return RANKS.index(rank) if rank in RANKS else len(RANKS)


def _same_value(a: SpecItem, b: SpecItem) -> bool:
    if a.number is not None and b.number is not None:
        return float(a.number) == float(b.number)
    return _norm(a.value) == _norm(b.value)


def resolve(items: list[SpecItem], family: str, numeric_only: bool = False, corroborate: set[float] | None = None
            ) -> tuple[list[SpecItem], list[str]]:
    """Ein Eintrag je Schluessel: hoechster Rang gewinnt; argparse-Defaults fallen weg, sobald eine hoehere Quelle den Schluessel
    setzt (D-Klasse). Widersprechen sich zwei Quellen GLEICHEN Rangs, wird nichts erzwungen (Hinweis statt Raten).
    corroborate (F4, code_grounded): Zahlen aus belegten Paper-Zitaten. Dann gilt zusaetzlich: eine Modell-Konfiguration schlaegt die
    allgemeine (train.conf), und von zwei widersprechenden gleichrangigen Werten gewinnt der, dessen Zahl das Paper nennt (wenn genau einer).
    Rueckgabe (Eintraege in Schluesselreihenfolge, Schluessel ohne Beleg)."""
    num, txt = keys_for(family, numeric_only)
    order = [k for k, _ in num + txt]
    out, missing = [], []
    for key in order:
        group = sorted((i for i in items if i.key == key), key=lambda i: _rank_index(i.rank))
        if not group:
            missing.append(key)
            continue
        if corroborate is not None and any(i.rank == "config" and i.specific for i in group):
            group = [i for i in group if not (i.rank == "config" and not i.specific)]
        win = group[0]
        alts = [i for i in group[1:] if not _same_value(i, win)]
        tie = [a for a in alts if a.rank == win.rank]
        note = ""
        if tie and corroborate is not None:
            cands = [win] + tie
            backed = [c for c in cands if c.number is not None and any(abs(float(c.number) - x) <= 1e-12 * max(1.0, abs(x)) for x in corroborate)]
            if backed and len({float(c.number) for c in backed}) == 1:
                win = backed[0]
                alts = [i for i in group if i is not win and not _same_value(i, win)]
                tie = []   # entschieden: der unterlegene gleichrangige Wert bleibt als Alternative im Dossier
                note = "gleichrangige Quellen widersprechen sich; entschieden durch die Zahl im Paper-Zitat"
        win.alternatives = [{"value": a.value, "source": a.source, "rank": a.rank, "quote": a.quote} for a in alts]
        if tie:
            note = "gleichrangige Quellen widersprechen sich: " + "; ".join(f"{a.value} ({a.source})" for a in tie)
            win.tie = [{"value": i.value, "source": i.source} for i in [win] + tie]
        win.note = note
        win.enforce = win.number is not None and not tie
        out.append(win)
    return out, missing


# ---- code_grounded: roher Ausschnitt ---------------------------------------------------------------------------------------
SECTION_RE = re.compile(r"^=== (.+?) ===$", re.M)
SH_FLAG_RE = re.compile(r"(?<![\w-])--([A-Za-z_][\w-]*)")
ADD_ARG_RE = re.compile(r"add_argument\((?:[^()]|\([^()]*\))*\)", re.S)
DEFAULT_KW_RE = re.compile(r",?[ \t]*\bdefault\s*=\s*(?:'[^']*'|\"[^\"]*\"|\[[^\]]*\]|\([^)]*\)|[^,)\n]+)")


def drop_run_script_defaults(excerpt: str) -> tuple[str, list[str]]:
    """Setzt ein Startskript (.sh) ein Flag, faellt der argparse-Default dieses Flags aus den Python-Stuecken weg (D-Klasse: der Default
    0.0001 neben `--learning_rate 0.005` im Skript). Rueckgabe (Ausschnitt, entfernte Flags). Deterministisch, ohne LLM."""
    heads = list(SECTION_RE.finditer(excerpt))
    sections = [(m.group(1), excerpt[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(excerpt)]) for i, m in enumerate(heads)]
    is_sh = lambda name: name.split(" (")[0].split("@")[0].endswith(".sh")   # Name wie "repo:scripts/etth1.sh@0c11366"
    flags = {f for name, body in sections if is_sh(name) for f in SH_FLAG_RE.findall(body)}
    if not flags:
        return excerpt, []
    dropped: list[str] = []

    def strip(call: re.Match) -> str:
        text = call.group(0)
        named = {f for f in flags if re.search(rf"['\"]--{re.escape(f)}['\"]", text)}
        if not named or not DEFAULT_KW_RE.search(text):
            return text
        dropped.extend(sorted(named))
        return DEFAULT_KW_RE.sub("", text, count=1)

    out = []
    for m, (name, body) in zip(heads, sections):
        if not is_sh(name):
            body = ADD_ARG_RE.sub(strip, body)
        out.append(f"=== {name} ===" + body)
    return excerpt[: heads[0].start()] + "".join(out), sorted(set(dropped))


def _assemble(model: str, family: str, kept: list[SpecItem], docs: list[Doc], names: list[str], details: list[SetupDetail]
              ) -> tuple[list[SpecItem], list[str]]:
    """code_grounded: zu den Eintraegen des Extraktors kommen die deterministisch gelesenen Zeilen des Ausschnitts (F1, F4 c); die Rangregel
    entscheidet (Startskript > Config > Paper), bei gleichem Rang gilt F4 (Modell-Konfiguration, Paper-Zahl)."""
    raw = raw_items(docs, names, model, family)
    return resolve(kept + raw, family, True, corroborate=paper_numbers(details, kept))


def build_spec(ask, details: list[SetupDetail], docs: list[Doc], claim_text: str, dataset: str, numeric_only: bool = False) -> Spec:
    """ask(system, user) -> Antworttext. Ein LLM-Aufruf; alles danach ist deterministisch.
    numeric_only (code_grounded): nur Zahlenschluessel (S2 erzwingt sie), der Ausschnitt bleibt roh im Spec (argparse-Defaults
    entfernt, wo ein Startskript das Flag setzt) und geht in Plan-, Code- und Reparatur-Prompt. Nur dort gelten F1 und F4:
    Quellen ohne @commit werden abgebildet, Startskript- und Konfigurationswerte deterministisch gelesen, Konflikte gleichen Rangs nach F4 entschieden."""
    model, family = model_name_of(claim_text), family_for(dataset)
    excerpt, names = focused_excerpt(docs, model, dataset, details)
    dropped_defaults: list[str] = []
    if numeric_only:
        excerpt, dropped_defaults = drop_run_script_defaults(excerpt)
    raw = _parse_list(ask(SPEC_SYSTEM, build_prompt(details, excerpt, claim_text, dataset, model, family, numeric_only)))
    kept, dropped = verify_entries(raw, docs, model, family, numeric_only, normalize=numeric_only)
    if numeric_only:
        items, missing = _assemble(model, family, kept, docs, names, details)
    else:
        items, missing = resolve(kept, family, numeric_only)
    return Spec(model=model, family=family, items=items, dropped=dropped, excerpt_docs=names, missing=missing,
                excerpt=excerpt if numeric_only else "", dropped_defaults=dropped_defaults)


def excerpt_docs(excerpt: str) -> list[Doc]:
    """Dokumente aus einem gespeicherten Ausschnitt (Abschnitte `=== name ===`); Namen ohne ' (function body)', gleiche Namen zusammengefuegt."""
    heads = list(SECTION_RE.finditer(excerpt))
    merged: dict[str, str] = {}
    for i, m in enumerate(heads):
        name = m.group(1).replace(" (function body)", "")
        merged[name] = merged.get(name, "") + excerpt[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(excerpt)]
    return [Doc(n, t) for n, t in merged.items()]


def rebuild_spec(old: dict, details: list[SetupDetail], paper_text: str | None = None) -> Spec:
    """Wiederholt den deterministischen Teil (F1, F4) auf einem gespeicherten Steckbrief (spec.json eines frueheren Laufs), OHNE neuen
    Extraktor-Aufruf: die Eintraege des Extraktors (Sieger und unterlegene Alternativen) und die verworfenen Eintraege mit Grund 'Quelle unbekannt'
    (diese werden gegen die Dokumente des Ausschnitts neu geprueft) gehen mit den neu gelesenen Zeilen des Ausschnitts durch dieselbe Rangregel.
    Alles, was schon geprueft war, bleibt geprueft (die Pruefregeln sind nur gelockert, nie verschaerft)."""
    model, family = old["model"], old["family"]
    docs = excerpt_docs(old["excerpt"]) + ([Doc("paper", paper_text)] if paper_text else [])
    kept: list[SpecItem] = []
    for it in old["items"]:
        base = {k: v for k, v in it.items() if k in SpecItem.__dataclass_fields__}
        sp = SpecItem(**{**base, "alternatives": [], "note": "", "tie": [], "enforce": False})
        if sp.origin == "raw":
            continue   # wird unten neu gelesen
        kept.append(sp)
        for a in it.get("alternatives", []):
            kept.append(SpecItem(it["key"], a["value"], a["quote"], a["source"], a["rank"], parse_number(it["key"], str(a["value"])),
                                 specific=_is_specific(a["source"], model)))
    redo = [d["item"] for d in old.get("dropped", []) if d.get("reason") == "Quelle unbekannt" and isinstance(d.get("item"), dict)]
    again, still = verify_entries(redo, docs, model, family, True, normalize=True)
    dropped = [d for d in old.get("dropped", []) if d.get("reason") != "Quelle unbekannt"] + still
    names = [d.name for d in docs if d.name != "paper"]
    items, missing = _assemble(model, family, kept + again, docs, names, details)
    return Spec(model=model, family=family, items=items, dropped=dropped, excerpt_docs=old.get("excerpt_docs", []), missing=missing,
                excerpt=old["excerpt"], dropped_defaults=old.get("dropped_defaults", []))


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
        if i.tie and spec.excerpt:   # code_grounded: Quellen gleichen Rangs widersprechen sich, kein Wert wird vorgegeben
            vals = "; ".join(f"{t['value']} [{_doc_path(t['source'])}]" for t in i.tie)
            lines.append(f"- {i.key}: sources of equal rank disagree ({vals}); not fixed here, decide with the paper and the code excerpt below")
            continue
        val = _fmt_number(i.number) if i.enforce else i.value
        lines.append(f"- {i.key} = {val}   [{i.rank}: \"{_short(i.quote)}\"]")
    if spec.missing and spec.excerpt:   # F4 (b): im Ausschnitt kann der Wert trotzdem stehen
        missing = ("\nNot extracted into this spec (if the authors' code excerpt below defines one of them, use that value; otherwise choose "
                   f"a standard value and say so in a comment): {', '.join(spec.missing)}.")
    elif spec.missing:
        missing = f"\nNot stated in any source (choose a standard value and say so in a comment): {', '.join(spec.missing)}."
    else:
        missing = ""
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


CG_INTRO = (
    "Authors' code for {model}: verbatim excerpt of the official repository (model file, layers it imports, preprocessing and "
    "learning-rate-schedule functions, run scripts, configs). It defines the method AND the training protocol of the paper's "
    "experiment: model definition, preprocessing, optimiser, learning-rate schedule, early stopping, how the dataset is run. "
    "Re-implement exactly this method in your own single script for the task below (our data file and output contract); do not "
    "copy data-loading or I/O code. If a script or default selects a model other than {model}, ignore that selection. "
    "Argparse defaults that a run script overrides were removed from the excerpt."
)


def format_code_grounded(spec: Spec, details: list[SetupDetail]) -> str:
    """code_grounded: numerischer Steckbrief (erzwungen, S2), belegte Paper-Angaben und der ROHE Repo-Ausschnitt statt destillierter Zitate."""
    from .setup_details import format_for_prompt_v2

    paper = format_for_prompt_v2([d for d in details if d.source == "paper"])
    parts = [format_spec(spec) if spec.items else "", paper]
    if spec.excerpt:
        parts.append(CG_INTRO.format(model=spec.model) + "\n\n" + spec.excerpt)
    return "\n\n".join(p for p in parts if p)


def as_dict(spec: Spec) -> dict:
    from dataclasses import asdict
    d = asdict(spec)
    d.pop("excerpt", None)   # roh nur im Prompt; Dossier traegt Stuecknamen und Laenge
    if not d.get("dropped_defaults"):
        d.pop("dropped_defaults", None)
    if spec.excerpt:
        d["excerpt_chars"] = len(spec.excerpt)
    return d

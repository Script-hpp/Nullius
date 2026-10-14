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
MODEL_SEL_RE = re.compile(r"(--)?\bmodel(?:_name|_type)?\b(\s*(?:={1,2}|:)\s*|\s+)['\"]?([A-Za-z][\w-]*)", re.I)
# Woerter hinter "model" in Fliesstext ("We train each model with a maximum 100 epochs", "each model w/ dropout"): nie ein Modellname
MODEL_PROSE_WORDS = frozenset(
    "a an the this that these those each every all any both with without w and or but nor is are was were be been being has have had do does did "
    "for on in at of to from by as if than then so it its we our us they their can could will would shall should may might must not no "
    "per via using use used uses trained train training tested test evaluated achieves achieved performs performed gets got reaches reached "
    "converges converged selected chosen based type name size selection size weights parameters params layers".split())
BINARY_EXT = re.compile(r"\.(pkl|pickle|npz|npy|pt|pth|bin|h5|hdf5|zip|gz|tar|png|jpe?g|gif|pdf|so|pyc|ckpt|mat|csv|tsv|parquet)$", re.I)
DATA_DIR = re.compile(r"(^|/)(data|dataset|datasets|checkpoints?|logs?|results?|figures?|pics?|img|images)/", re.I)


def keys_for(family: str, numeric_only: bool = False) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """numeric_only (code_grounded): nur Zahlenschluessel; was nicht per CONFIG erzwungen wird, steht roh im Code-Ausschnitt."""
    if numeric_only:
        return NUMERIC_COMMON + NUMERIC_FAMILY.get(family, []), []
    return NUMERIC_COMMON + NUMERIC_FAMILY.get(family, []), TEXT_COMMON + TEXT_FAMILY.get(family, [])


def model_name_of(claim_text: str) -> str:
    """Modellname des Claims: Text vor ' erreicht ' (papers.yaml), ohne Variante nach '/' (PatchTST/42 -> PatchTST) und ohne Klammerzusatz (GCN (this paper) -> GCN)."""
    head = claim_text.split(" erreicht ")[0].strip() if " erreicht " in claim_text else claim_text.split()[0]
    head = re.sub(r"\s*\([^)]*\)\s*$", "", head)   # Katalog-Methode aus der Tabelle: "GCN (this paper)" -> GCN
    return head.split("/")[0].strip()


# ---- Quelltext-Filter (S4) ------------------------------------------------------------------------------------------
def is_source_text(path: str, text: str) -> bool:
    """Nur lesbarer Quelltext/Konfiguration: keine Binaer- oder Datendateien (SGC: sechs Pickles, ~153 000 Zeichen)."""
    if BINARY_EXT.search(path) or DATA_DIR.search(path) and not path.lower().endswith('.py'):
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


def model_files(docs: list[Doc], model: str, entry_fallback: bool = True) -> list[Doc]:
    """Repo-Dateien, die die Klasse des Claim-Modells definieren oder so heissen (models/NLinear.py, models/dgi.py).

    entry_fallback: Ist die Methode keine Modellklasse (Trainingsverfahren wie BANGS 2410.09348: Self-Training mit
    Knotenauswahl, kein `class BANGS`), gilt das Einstiegsskript, das die Startbefehle der Autoren am haeufigsten
    aufrufen, als Kern; seine lokalen Importe folgen wie beim Modell. Das Repo ist immer das gepruefte offizielle."""
    m = model.lower()
    cls = re.compile(rf"^\s*class\s+{re.escape(model)}\b", re.M | re.I)
    out = [d for d in docs if d.name.startswith("repo:") and _path(d).endswith(".py") and (_stem(_path(d)) == m or cls.search(d.text))]
    if not out and entry_fallback:
        return method_entry_files(docs)
    return sorted(out, key=lambda d: (_stem(_path(d)) != m, not re.search(r'(^|/)models?/', _path(d), re.I), len(d.text)))


def method_entry_files(docs: list[Doc]) -> list[Doc]:
    """Das Python-Skript, das Startbefehle in Skripten/README am haeufigsten aufrufen (auch auskommentierte Beispiele);
    bei Gleichstand keines, damit nie geraten wird."""
    py = {_path(d): d for d in docs if d.name.startswith("repo:") and _path(d).endswith(".py")}
    counts: dict[str, int] = {}
    for d in docs:
        if not (d.name.startswith("repo:") and re.search(r"\.(sh|md|txt|rst)$", _path(d), re.I) or d.name == "repo-readme"):
            continue
        for m in re.finditer(r"\bpython(?:3(?:\.\d+)?)?\b[^\n]*?\s\.?/?([\w./-]+\.py)\b", d.text):   # auch `-W ignore`
            if m.group(1) in py:
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return []
    return [py[ranked[0][0]]]


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
    model_code_present: bool = True
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
        flag, sep, name = m.group(1), m.group(2), m.group(3)
        if name.lower() in MODEL_PROSE_WORDS or len(name) < 2:
            continue
        if not flag and not re.search(r"[=:]", sep) and not (re.search(r"[A-Z0-9_]", name) or "-" in name):
            continue   # "model gcn" im Fliesstext ohne Zuweisung/Flag: nur Namen mit Grossbuchstabe/Ziffer ("model DLinear") zaehlen
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
FORECAST_HORIZON = 96   # alle ETT-Tasks prognostizieren 96 Schritte (harness/ett_data.PRED_LEN)

# ---- Startskripte je Aufruf (E08/E09): nur der Aufruf des Experiments zaehlt, nicht der fuer einen anderen Datensatz/Horizont ------
SH_CONT_RE = re.compile(r"\\[ \t]*\r?\n")
SH_VALUE_RE = re.compile(r"--([A-Za-z_][\w-]*)(?:[ \t]+|=)(['\"]?)([^\s'\"]+)\2")
DATA_FLAGS = {"data", "data_path", "dataset", "data_name", "dataset_name"}
HORIZON_FLAGS = {"pred_len", "horizon", "prediction_length", "out_len", "output_len", "forecast_horizon"}
MODEL_FLAGS = {"model", "model_name", "model_type"}
FEATURE_FLAGS = {'features', 'feature'}  # argparse permits the unambiguous --feature abbreviation


def _flag(name: str) -> str:
    return name.lower().replace("-", "_")


def sh_invocations(body: str) -> tuple[list[str], list[str]]:
    """(Aufrufe, uebrige Zeilen) eines Shell-Skripts. Fortsetzungszeilen (`\\`) werden zu einem Aufruf zusammengefuegt; ein Aufruf ist eine
    Zeile mit mindestens einem `--flag`."""
    calls, rest = [], []
    for line in SH_CONT_RE.sub(" ", body).splitlines():
        (calls if SH_FLAG_RE.search(line) else rest).append(line)
    return calls, rest


def _dataset_token(value: str) -> str:
    stem = value.rsplit("/", 1)[-1]
    stem = re.sub(r"\.(csv|npz|npy|txt|pkl|h5)$", "", stem, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def call_matches(call: str, dataset: str | None = None, horizon: int | None = None, model: str | None = None,
                 features: str | None = None) -> bool:
    """Passt ein Aufruf zum Experiment des Claims? Nur ein ausdruecklich anderer Wert (Literal, keine Shell-Variable) schliesst ihn aus."""
    values: dict[str, list[str]] = {}
    for name, _, value in SH_VALUE_RE.findall(call):
        values.setdefault(_flag(name), []).append(value)

    def literals(flags: set[str]) -> list[str]:
        return [v for f in flags for v in values.get(f, []) if not v.startswith("$")]

    if dataset and (lits := literals(DATA_FLAGS)):
        want = _dataset_token(dataset)
        # Praefix genuegt ("cora" im Skript fuer den Task Cora-LCC); ETTh1 und ETTh2 bleiben verschieden
        if not any(t == want or (min(len(t), len(want)) >= 4 and (t.startswith(want) or want.startswith(t)))
                   for t in map(_dataset_token, lits)):
            return False
    if horizon is not None and (lits := literals(HORIZON_FLAGS)):
        try:
            if all(float(v) != float(horizon) for v in lits):
                return False
        except ValueError:
            pass
    if model and (lits := literals(MODEL_FLAGS)) and model.lower() not in {v.lower() for v in lits}:
        return False
    if features and (lits := literals(FEATURE_FLAGS)) and any(v.upper() != features for v in lits):
        return False
    return True


def _doc_path(name: str) -> str:
    return name.split(":", 1)[-1].split("@")[0].replace(" (function body)", "")


def _is_specific(source: str, model: str) -> bool:
    """Konfigurationsdatei mit dem Modellnamen im Pfad (gat.conf.yaml) gilt vor der allgemeinen (train.conf.yaml). Andere Quellen: immer."""
    path = _doc_path(source)
    return model.lower() in path.lower() if CONFIG_RE.search(path) else True


def raw_items(docs: list[Doc], names: list[str], model: str, family: str, dataset: str | None = None,
              horizon: int | None = None) -> list[SpecItem]:
    """F1 (Startskript) und F4 (c, Konfiguration): Zuweisungen `name = Zahl`, `name: Zahl` und `--flag Zahl` in den Startskripten und
    Konfigurationsdateien des Ausschnitts. Steht ein Schluessel mit mehreren verschiedenen Werten da, wird er nicht gelesen.
    Shell-Skripte (E09): `--flag Zahl` nur aus den Aufrufen, die zu Datensatz, Horizont und Modell passen (`call_matches`); ein anderer
    Aufruf im selben Skript verdraengt den passenden Wert nicht mehr. Deterministisch, kein LLM."""
    num_keys = {k for k, _ in keys_for(family, True)[0]}
    want = {n.replace(" (function body)", "") for n in names}
    out: list[SpecItem] = []
    for d in docs:
        path = _doc_path(d.name)
        if d.name not in want or not (START_RE.search(path) or CONFIG_RE.search(path)):
            continue
        found: dict[str, list[tuple[str, str]]] = {}
        if path.endswith(".sh"):
            calls, rest = sh_invocations(d.text)
            from .experiment_settings import shell_calls
            identities = {key: model for key in MODEL_FLAGS} if model else {}
            if dataset: identities |= {key: dataset for key in DATA_FLAGS}
            if horizon is not None: identities |= {key: horizon for key in HORIZON_FLAGS}
            if family == 'forecast': identities |= {key: 'M' for key in FEATURE_FLAGS}
            lines = [(ln, False) for ln in rest] + [(c, True) for _, c, _ in shell_calls(d.text, identities)
                     if call_matches(c, dataset, horizon, model, 'M' if family == 'forecast' else None)]
        else:
            lines = [(ln, False) for ln in d.text.splitlines()]
        for line, is_call in lines:
            m = ASSIGN_RE.match(line.rstrip())
            hits = [(m.group(1), m.group(2), line.strip())] if m else []
            if is_call:   # Zitat = das Flag selbst (der zusammengefuegte Aufruf steht so nicht woertlich im Skript)
                hits += [(f.group(1), f.group(2), f.group(0)) for f in FLAG_NUM_RE.finditer(line)]
            for name, value, quote in hits:
                key = ALIAS_KEY.get(name.lower().replace("-", "_")) or ALIAS_KEY.get(name.lower())
                if key in num_keys:
                    found.setdefault(key, []).append((value, quote))
        for key, occ in found.items():
            if len({float(v) for v, _ in occ}) != 1:
                continue
            value, quote = occ[0]
            number = parse_number(key, value)
            if number is None:
                continue
            out.append(SpecItem(key, value, quote, d.name, source_rank(d.name, quote), number, specific=_is_specific(d.name, model), origin="raw"))
    return out


# E12: Bestaetigung nur fuer denselben Parameter. Woerter, an denen ein Paper-Zitat seinen Parameter erkennen laesst ("horizon" ist kein
# Steckbrief-Schluessel, bindet aber Zahlen wie 96 oder 32, damit sie keinen fremden Konflikt entscheiden).
PARAM_WORDS = {
    "lr": r"learning[ _-]?rates?|\blr\b",
    "weight_decay": r"weight[ _-]?decay|\bl2\b|regulari[sz]",
    "epochs": r"\bepochs?\b",
    "batch_size": r"\bbatch",
    "dropout": r"dropout",
    "hidden": r"hidden|\bunits\b|d_model|embedding (?:size|dim)",
    "layers": r"\blayers?\b",
    "patience": r"patience|early[ -]?stop",
    "heads": r"\bheads?\b",
    "seq_len": r"look-?back|input (?:length|window|sequence)|seq(?:uence)?[ _]len|history length|lookback",
    "horizon": r"horizon|prediction length|pred(?:iction)?[ _]len|forecast(?:ing)? (?:length|steps)|output length",
}
PARAM_WORDS_RE = {k: re.compile(v, re.I) for k, v in PARAM_WORDS.items()}
UNBOUND = "*"


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for tok in NUM_TOKEN_RE.findall(text):
        try:
            out.add(float(tok))
        except ValueError:
            pass
    return out


def quote_params(text: str) -> set[str]:
    """Parameter, die ein Text ausdruecklich nennt (Schluessel des Steckbriefs oder 'horizon')."""
    return {k for k, rx in PARAM_WORDS_RE.items() if rx.search(text or "")}


def paper_numbers(details: list[SetupDetail], items: list[SpecItem]) -> dict[str, set[float]]:
    """Zahlen aus belegten Paper-Zitaten (Rolle 4 und Steckbrief-Eintraege der Quelle 'paper'), je Parameter (E12).
    Ein Zitat, das Parameter nennt, stuetzt nur diese; ein Zitat ohne Parameterwort (Tabellenzeile) bleibt ungebunden (UNBOUND) und
    stuetzt jeden Schluessel. Das Thema einer Angabe bindet ihren Wert. Steckbrief-Eintraege des Papers stuetzen ihren eigenen Schluessel."""
    out: dict[str, set[float]] = {UNBOUND: set()}
    for d in details:
        if getattr(d, "source", "") != "paper" or getattr(d, "excluded", ""):
            continue
        for k in quote_params(d.quote) or {UNBOUND}:   # Tabellenzeile ohne Parameterwort: ungebunden
            out.setdefault(k, set()).update(_numbers(d.quote))
        for k in quote_params(getattr(d, "topic", "")):   # das Thema bindet nur den eigenen Wert der Angabe
            out.setdefault(k, set()).update(_numbers(str(getattr(d, "value", ""))))
    for i in items:
        if i.source == "paper":
            out.setdefault(i.key, set()).update(_numbers(i.quote))
    return out


def _corroborating(corroborate: set[float] | dict[str, set[float]], key: str) -> set[float]:
    """Zahlen, die einen Wert fuer `key` stuetzen. Eine flache Menge (aeltere Aufrufer) gilt fuer jeden Schluessel."""
    if isinstance(corroborate, dict):
        return set(corroborate.get(key, set())) | set(corroborate.get(UNBOUND, set()))
    return set(corroborate)


def normalize_source(source: str, names: list[str]) -> str:
    """F4 (a): Quelle ohne `@commit` (oder ohne `repo:`) auf die eindeutige Dokumentzeile abbilden. Unbekannt oder mehrdeutig: unveraendert."""
    if source in names:
        return source
    base = source.removeprefix("repo:").split("@")[0]
    hits = [n for n in names if n.startswith("repo:") and n.removeprefix("repo:").split("@")[0] == base]
    return hits[0] if len(hits) == 1 else source


# E07: Zahl mit anderer Rolle. "inference speed with batch size 32" belegt keine Trainingsbatchgroesse. Ein Zitat mit Inferenz-/Effizienzkontext
# zaehlt nur, wenn es das Training ausdruecklich nennt ("All models are trained for 10 epochs ... inference speed").
INFERENCE_CONTEXT_RE = re.compile(
    r"\binferen\w*|throughput|latency|\bspeed\b|efficien\w*|run-?time|wall-?clock|\bFLOPs?\b|memory (?:usage|footprint|consumption|cost)",
    re.I)
TRAINING_CONTEXT_RE = re.compile(r"\btrain\w*|fine-?tun\w*|optimi[sz]\w*", re.I)


def not_training_context(quote: str) -> bool:
    return bool(INFERENCE_CONTEXT_RE.search(quote)) and not TRAINING_CONTEXT_RE.search(quote)


def verify_entries(raw: list, docs: list[Doc], model: str, family: str, numeric_only: bool = False, normalize: bool = False
                   ) -> tuple[list[SpecItem], list[dict]]:
    """Deterministisch: bekannter Schluessel, bekannte Quelle, Zitat im Dokument, Zahl im Zitat, keine fremde Modellwahl.
    normalize (F4 a, code_grounded): fehlt der `@commit` in der Quelle, wird sie auf die eindeutige Dokumentzeile abgebildet."""
    num, txt = keys_for(family, numeric_only)
    num_keys, txt_keys = {k for k, _ in num}, {k for k, _ in txt}
    from .experiment_settings import active_paper_text
    hay = {d.name: _norm(active_paper_text(d.text) if d.name == 'paper' else d.text) for d in docs}
    kept, dropped = [], []
    for it in raw:
        if not isinstance(it, dict):
            dropped.append({"item": it, "reason": "kein Objekt"})
            continue
        key, value, quote, source = (str(it.get(k, "")).strip() for k in ("key", "value", "quote", "source"))
        source = re.sub(r"\s*\(function body\)\s*$", "", source)
        if normalize:
            source = normalize_source(source, list(hay))
            if source == 'discovery' and quote_in_text(quote, hay.get('paper', '')):
                source = 'paper'  # discovery is provenance; the actual quoted document is the paper
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
        elif key in num_keys and not_training_context(quote):
            reason = "Zitat beschreibt Inferenz/Effizienz, nicht das Training"
        elif key == 'seq_len' and ambiguous_length_quote(quote):
            reason = "Zitat nennt mehrere Eingabelaengen, keine einzelne Experimentvorgabe"
        elif source.startswith('repo:') and overridden_signature_default(quote, value, source, docs):
            reason = "Default der Funktionssignatur; der Aufruf im Repo uebergibt diesen Parameter selbst"
        if reason:
            dropped.append({"item": it, "reason": reason})
            continue
        number = parse_number(key, value) if key in num_keys else None
        kept.append(SpecItem(key, value, quote, source, source_rank(source, quote), number, specific=_is_specific(source, model)))
    return kept, dropped


def overridden_signature_default(quote: str, value: str, source: str, docs: list[Doc]) -> bool:
    """`def __init__(self, patience=7)` is not the run's value when a call site passes `patience=...` itself.
    RLinear live 2026-10-14: EarlyStopping's signature default 7 outranked the `--patience` default 10 that
    `EarlyStopping(patience=self.args.patience)` actually receives; the same utils/tools.py is in most LTSF repos."""
    sig = re.search(r"\bdef\s+(\w+)\s*\(([^)]*)\)?", quote)
    if not sig:
        return False
    params = [p for p, v in re.findall(r"(\w+)\s*=\s*([^,)\s]+)", sig.group(2)) if _number_in_quote(value, v) or v == value]
    if not params:
        return False
    callee = sig.group(1)
    if callee == '__init__':
        text = next((d.text for d in docs if d.name == source), '')
        at = text.find(sig.group())
        classes = re.findall(r"^\s*class\s+(\w+)", text[:at] if at >= 0 else '', re.M)
        if not classes:
            return False
        callee = classes[-1]
    calls = [re.compile(rf"(?<![\w.]def ){re.escape(callee)}\s*\((?:[^()]|\([^()]*\))*?\b{re.escape(p)}\s*=") for p in params]
    return any(c.search(d.text) for d in docs if d.name.startswith('repo:') for c in calls)


def ambiguous_length_quote(quote: str) -> bool:
    # A multi-column option header is not a selected setting. Preserve sentences
    # that state input length and a separate prediction horizon.
    columns = re.findall(r'\\multicolumn\{\d+\}\{[^}]*\}\{([\d.]+)\}', quote)
    if len(set(columns)) > 1:
        return True
    return bool(re.search(r'(?:search|choices?|candidates?|range|lengths|windows)\b[^.\n]*\[?\d+(?:\s*[,/]\s*\d+){1,}', quote, re.I))


def _rank_index(rank: str) -> int:
    return RANKS.index(rank) if rank in RANKS else len(RANKS)


def _same_value(a: SpecItem, b: SpecItem) -> bool:
    if a.number is not None and b.number is not None:
        return float(a.number) == float(b.number)
    return _norm(a.value) == _norm(b.value)


def resolve(items: list[SpecItem], family: str, numeric_only: bool = False,
            corroborate: set[float] | dict[str, set[float]] | None = None) -> tuple[list[SpecItem], list[str]]:
    """Ein Eintrag je Schluessel: hoechster Rang gewinnt; argparse-Defaults fallen weg, sobald eine hoehere Quelle den Schluessel
    setzt (D-Klasse). Widersprechen sich zwei Quellen GLEICHEN Rangs, wird nichts erzwungen (Hinweis statt Raten).
    corroborate (F4, code_grounded): Zahlen aus belegten Paper-Zitaten, je Parameter (paper_numbers; E12). Dann gilt zusaetzlich: eine Modell-Konfiguration schlaegt die
    allgemeine (train.conf), und von zwei widersprechenden gleichrangigen Werten gewinnt der, dessen Zahl das Paper nennt (wenn genau einer).
    Rueckgabe (Eintraege in Schluesselreihenfolge, Schluessel ohne Beleg)."""
    num, txt = keys_for(family, numeric_only)
    order = [k for k, _ in num + txt]
    order += sorted({i.key for i in items if i.origin == 'runtime' and i.key not in order})
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
            support = _corroborating(corroborate, key)
            backed = [c for c in cands if c.number is not None and any(abs(float(c.number) - x) <= 1e-12 * max(1.0, abs(x)) for x in support)]
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


def drop_run_script_defaults(excerpt: str, dataset: str | None = None, horizon: int | None = None,
                             model: str | None = None) -> tuple[str, list[str]]:
    """Setzt ein Startskript (.sh) ein Flag, faellt der argparse-Default dieses Flags aus den Python-Stuecken weg (D-Klasse: der Default
    0.0001 neben `--learning_rate 0.005` im Skript). Rueckgabe (Ausschnitt, entfernte Flags). Deterministisch, ohne LLM.
    E08: Mit Datensatz/Horizont/Modell zaehlen nur die Aufrufe dieses Experiments (`call_matches`). Ein Flag, das nur ein Aufruf fuer
    einen anderen Datensatz oder Horizont setzt, laesst den Default stehen, denn der passende Aufruf benutzt ihn."""
    heads = list(SECTION_RE.finditer(excerpt))
    sections = [(m.group(1), excerpt[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(excerpt)]) for i, m in enumerate(heads)]
    is_sh = lambda name: name.split(" (")[0].split("@")[0].endswith(".sh")   # Name wie "repo:scripts/etth1.sh@0c11366"
    from .experiment_settings import shell_calls
    identities = {key: horizon for key in HORIZON_FLAGS if horizon is not None}
    identities |= {key: model for key in MODEL_FLAGS if model}
    identities |= {key: dataset for key in DATA_FLAGS if dataset}
    if horizon is not None:
        identities |= {key: 'M' for key in FEATURE_FLAGS}
    flags = {f for name, body in sections if is_sh(name) for _, call, _ in shell_calls(body, identities)
             if call_matches(call, dataset, horizon, model, 'M' if horizon is not None else None)
             for f in SH_FLAG_RE.findall(call)}
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


# N04: Mechanismus-Schluessel, die nur gelten, wenn der Modellcode des Autors den Mechanismus enthaelt. Ein globaler argparse-Default
# (dropout 0.1 in run.py) fuer ein Modell ohne Dropout darf keine Reparatur erzwingen, die eine Dropout-Schicht einbaut.
MECHANISM_WORDS = {"dropout": re.compile(r"dropout", re.I)}


def model_core_docs(docs: list[Doc], model: str) -> list[Doc]:
    """Modelldatei(en) des Claims und alle gelesenen lokalen Module, die sie (transitiv) importieren."""
    from .repo_source import imported_files

    repo = {_path(d): d for d in docs if d.name.startswith("repo:")}
    core = {_path(d): d for d in model_files(docs, model)[:2]}
    while True:
        new = [p for p in imported_files({p: d.text for p, d in core.items()}, list(repo), relative=True) if p not in core]
        if not new:
            return list(core.values())
        core.update({p: repo[p] for p in new})


def inactive_mechanisms(items: list[SpecItem], docs: list[Doc], model: str) -> tuple[list[SpecItem], list[dict]]:
    """Eintraege fuer Mechanismen, die der gelesene Modellkern nicht enthaelt, fallen aus dem Steckbrief (mit Grund im Dossier)."""
    core = model_core_docs(docs, model)
    if not core:
        return items, []
    text = "\n".join(d.text for d in core)
    keep, dropped = [], []
    for i in items:
        rx = MECHANISM_WORDS.get(i.key)
        if rx is not None and not rx.search(text):
            dropped.append({"item": {"key": i.key, "value": i.value, "quote": i.quote, "source": i.source},
                            "reason": "Modellcode des Autors enthaelt diesen Mechanismus nicht"})
        else:
            keep.append(i)
    return keep, dropped


def _assemble(model: str, family: str, kept: list[SpecItem], docs: list[Doc], names: list[str], details: list[SetupDetail],
              dataset: str | None = None) -> tuple[list[SpecItem], list[str], list[dict]]:
    """code_grounded: zu den Eintraegen des Extraktors kommen die deterministisch gelesenen Zeilen des Ausschnitts (F1, F4 c); die Rangregel
    entscheidet (Startskript > Config > Paper), bei gleichem Rang gilt F4 (Modell-Konfiguration, Paper-Zahl). Startskripte zaehlen nur mit
    den Aufrufen dieses Experiments (E09); Mechanismen ohne Gegenstueck im Modellkern fallen weg (N04)."""
    horizon = FORECAST_HORIZON if family == "forecast" else None
    raw = raw_items(docs, names, model, family, dataset=dataset, horizon=horizon)
    # Preserve primary-source Discovery settings independently of model recall.
    discovery_raw = [{'key': k, 'value': d.value, 'quote': d.quote, 'source': d.source}
                     for d in details if d.source == 'discovery' and not getattr(d, 'excluded', '')
                     for k in quote_params(d.topic) if k in {key for key, _ in keys_for(family, True)[0]}]
    discovered, _ = verify_entries(discovery_raw, docs, model, family, True, normalize=True)
    for item in discovered:
        item.origin = 'source_detail'
    lengths = {i.number for i in discovered if i.key == 'seq_len'}
    runtime = runtime_items(docs, names, model, family, dataset, next(iter(lengths)) if len(lengths) == 1 else None)
    runtime_keys = {i.key for i in runtime}
    raw = [i for i in raw if i.key not in runtime_keys]
    kept = [i for i in kept if not (i.source.startswith('repo:') and (i.key in runtime_keys or _relabelled_flag(i, runtime_keys)))]
    for item in discovered:
        if item.key == 'seq_len':
            conflicts = [r for r in runtime if r.key == item.key and r.number != item.number]
            if conflicts:
                from .source_context import SourceContextBlocked
                raise SourceContextBlocked('Author launch input length conflicts with the selected source setting; no experiment was started')
    items, missing = resolve(kept + raw + discovered + runtime, family, True, corroborate=paper_numbers(details, kept))
    ambiguous = [i.key for i in items if i.tie and any(r.key == i.key for r in runtime)]
    if ambiguous:
        from .source_context import SourceContextBlocked
        raise SourceContextBlocked('Author launch settings remain ambiguous: ' + ', '.join(ambiguous) +
                                   '. No experiment was started')
    items, inactive = inactive_mechanisms(items, docs, model)
    return items, missing, inactive


def _relabelled_flag(item: SpecItem, runtime_keys: set[str]) -> bool:
    """A quoted CLI flag that the launch already resolves under its own name cannot also count as another setting.
    SARAF (2026-10-14): the extractor labelled `--n_period` as `layers`; both were enforced for one flag, the agent
    could satisfy only one, and the spec repairs looped until the job budget ran out."""
    flags = {ALIAS_KEY.get(_flag(f), _flag(f)) for f in re.findall(r"--([A-Za-z_][\w-]*)", item.quote)}
    return bool(flags & runtime_keys) and item.key not in flags


def runtime_items(docs: list[Doc], names: list[str], model: str, family: str, dataset: str | None, selected_length=None) -> list[SpecItem]:
    """Effective settings of matching launch commands, never executing the authors' code."""
    if family != 'forecast':
        return []
    import ast
    import math
    from .experiment_settings import shell_calls, entrypoint, python_settings, namespace_reads
    repo = {_path(d): d for d in docs if d.name.startswith('repo:')}
    active = set()
    for doc in model_core_docs(docs, model):
        try: tree = ast.parse(doc.text)
        except SyntaxError: continue
        active.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and
                      isinstance(n.value, ast.Name) and n.value.id in ('configs', 'config', 'args', 'cfg'))
    common = {k for k, _ in keys_for(family, True)[0]}
    architectural = {'hidden', 'layers', 'heads', 'dropout'}
    infrastructure = {'seed', 'gpu', 'devices', 'use_gpu', 'use_multi_gpu', 'num_workers', 'itr', 'is_training'}
    out = []
    incompatible_lengths = False
    for doc in docs:
        if doc.name not in names or not _path(doc).endswith('.sh'): continue
        identities = {key: FORECAST_HORIZON for key in HORIZON_FLAGS} | {key: model for key in MODEL_FLAGS}
        if dataset: identities |= {key: dataset for key in DATA_FLAGS}
        identities |= {key: 'M' for key in FEATURE_FLAGS}
        for original, expanded, bindings in shell_calls(doc.text, identities):
            if not call_matches(expanded, dataset, FORECAST_HORIZON, model, 'M'): continue
            script = repo.get(entrypoint(expanded))
            if script is None: continue
            try: script_tree = ast.parse(script.text)
            except SyntaxError: continue
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('parse_args', 'parse_known_args')
                       for n in ast.walk(script_tree)):
                continue  # no observed argparse namespace to interpret; retain existing source-literal rules
            values, evidence = python_settings(script.text, expanded, [d.text for p, d in repo.items() if p.endswith('.py') and d is not script])
            if any(key in evidence and key not in values for key in FEATURE_FLAGS | {'enc_in'}):
                from .source_context import SourceContextBlocked
                raise SourceContextBlocked('Author launch feature/channel identity is unresolved; no experiment was started')
            if any(values.get(key, 'M') != 'M' for key in FEATURE_FLAGS) or values.get('enc_in', 7) != 7:
                from .source_context import SourceContextBlocked
                raise SourceContextBlocked('Author launch feature/channel identity is incompatible; no experiment was started')
            if selected_length is not None and 'seq_len' in values and values['seq_len'] != selected_length:
                incompatible_lengths = True
                continue
            for identity, desired in [('model', model), ('model_name', model), ('model_type', model)]:
                if identity in evidence and str(values.get(identity, '')).lower() != model.lower():
                    from .source_context import SourceContextBlocked
                    raise SourceContextBlocked('Author launch model does not match the selected method; no experiment was started')
            for identity in HORIZON_FLAGS:
                if identity in evidence and values.get(identity) != FORECAST_HORIZON:
                    from .source_context import SourceContextBlocked
                    raise SourceContextBlocked('Author launch prediction horizon is unresolved or incompatible; no experiment was started')
            for identity in DATA_FLAGS - {'data_path'}:
                if dataset and identity in evidence and _dataset_token(str(values.get(identity, ''))) != _dataset_token(dataset):
                    from .source_context import SourceContextBlocked
                    raise SourceContextBlocked('Author launch dataset is unresolved or incompatible; no experiment was started')
            # Device and run-control flags (gpu = device_ids[0] in SARAF's run.py) never change a CPU measurement;
            # the seed is set by our harness. They must not block a launch as "model parameters".
            unresolved = sorted(name for name in set(evidence) - set(values) - infrastructure if name in active or
                                ALIAS_KEY.get(name.lower(), name) in common - architectural)
            if unresolved:
                from .source_context import SourceContextBlocked
                raise SourceContextBlocked('Author launch has unresolved model parameters: ' + ', '.join(unresolved) +
                                           '. No experiment was started')
            explicit = {_flag(n) for n, _, _ in SH_VALUE_RE.findall(expanded)} - infrastructure
            read = set().union(*(namespace_reads(d.text, values) for d in repo.values()))
            explicit &= read
            # Derived constructor values already include their source inputs.
            # Do not demand redundant CONFIG entries for intermediate CLI knobs.
            derived_inputs = {name for key in active if key in evidence and '\n' in evidence[key]
                              for name in re.findall(r'args\.(\w+)', evidence[key]) if name != key}
            for name, value in values.items():
                key = ALIAS_KEY.get(name.lower()) or name
                if key in architectural and name not in active: continue
                if key not in common and name not in active and (name not in explicit or name in derived_inputs): continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): continue
                source, quote = runtime_witness(doc, script, name, evidence.get(name, ''), original, bindings)
                out.append(SpecItem(key, str(value), quote, source,
                                    'start script', value, origin='runtime'))
    if incompatible_lengths and not out:
        from .source_context import SourceContextBlocked
        raise SourceContextBlocked('No author launch matches the selected input length; no experiment was started')
    return out


def runtime_witness(doc, script, name, evidence, original, bindings):
    """One actual source fragment for a derived setting, never a synthesized quote."""
    parts = re.split(r'\n(?=[A-Za-z_]\w*\.\w+\s*[+*/-]?=)', evidence)
    if len(parts) > 1:
        return script.name, parts[-1]  # the original post-parse arithmetic statement
    if quote_in_text(evidence, _norm(script.text)):
        return script.name, evidence
    if quote_in_text(evidence, _norm(doc.text)):
        return doc.name, evidence
    for flag, _, raw in SH_VALUE_RE.findall(original):
        if _flag(flag) != name.lower(): continue
        var = re.fullmatch(r'\$(?:\{(\w+)\}|(\w+))', raw)
        if var:
            bound = var[1] or var[2]
            for line in doc.text.splitlines():
                if re.match(rf'\s*(?:for\s+{re.escape(bound)}\s+in\b|(?:export\s+)?{re.escape(bound)}\s*=)', line):
                    return doc.name, line.strip()
    # Retain a real original flag fragment even when its value came from a binding.
    match = re.search(rf'--{re.escape(name)}(?:\s+|=)[^\s\\]+', doc.text)
    return (doc.name, match[0]) if match else (script.name, evidence)


def build_spec(ask, details: list[SetupDetail], docs: list[Doc], claim_text: str, dataset: str, numeric_only: bool = False,
               repo_listing: list[str] | None = None) -> Spec:
    """ask(system, user) -> Antworttext. Ein LLM-Aufruf; alles danach ist deterministisch.
    numeric_only (code_grounded): nur Zahlenschluessel (S2 erzwingt sie), der Ausschnitt bleibt roh im Spec (argparse-Defaults
    entfernt, wo ein Startskript das Flag setzt) und geht in Plan-, Code- und Reparatur-Prompt. Nur dort gelten F1 und F4:
    Quellen ohne @commit werden abgebildet, Startskript- und Konfigurationswerte deterministisch gelesen, Konflikte gleichen Rangs nach F4 entschieden."""
    model, family = model_name_of(claim_text), family_for(dataset)
    if numeric_only:
        from .source_context import author_excerpt
        excerpt, names = author_excerpt(docs, model, dataset, listing=repo_listing)
    else:
        excerpt, names = focused_excerpt(docs, model, dataset, details)
    dropped_defaults: list[str] = []
    if numeric_only:
        excerpt, dropped_defaults = drop_run_script_defaults(
            excerpt, dataset=dataset, horizon=FORECAST_HORIZON if family == "forecast" else None, model=model)
    raw = _parse_list(ask(SPEC_SYSTEM, build_prompt(details, excerpt, claim_text, dataset, model, family, numeric_only)))
    kept, dropped = verify_entries(raw, docs, model, family, numeric_only, normalize=numeric_only)
    if numeric_only:
        items, missing, inactive = _assemble(model, family, kept, docs, names, details, dataset=dataset)
        dropped += inactive
    else:
        items, missing = resolve(kept, family, numeric_only)
    return Spec(model=model, family=family, items=items, dropped=dropped, excerpt_docs=names, missing=missing,
                excerpt=excerpt if numeric_only else "", dropped_defaults=dropped_defaults,
                model_code_present=bool(model_files(docs, model)))


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
    items, missing, inactive = _assemble(model, family, kept + again, docs, names, details)
    dropped += inactive
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
    "treat every declared layer as active: trace the actual forward/encoder path and the functions it calls. "
    "Preserve its normalisation, retrieval exclusions and training protocol. Do not "
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

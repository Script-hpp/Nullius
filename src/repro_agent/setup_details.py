"""Tavily-Rolle 4: Setup-Details (Vorverarbeitung, Split, Hyperparameter, Training) aus dem Paper.

Allgemein gehalten: fragt nach ALLEN Angaben zum Versuchsaufbau, nicht nach einem bestimmten
Detail. Wie bei Rolle 2 gilt: die KI liest, ein Programm prueft. Jede Angabe muss mit einem
woertlichen Zitat belegt sein, das im Quelltext steht; sonst wird sie verworfen.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass

from .claims_source import _norm
from .llm import LLMClient

MAX_TEXT_CHARS = 60_000
CHUNK_CHARS = 5_000

SYSTEM = (
    "You extract experimental-setup facts from a research paper so someone else can reproduce a "
    "result. Keep reasoning brief."
)


@dataclass(frozen=True)
class Doc:
    name: str      # "paper" | "repo-readme" | "repo:<pfad>@<commit>"
    text: str


@dataclass(frozen=True)
class SetupDetail:
    topic: str
    value: str
    quote: str
    source: str = "paper"


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Absatzweise in Stuecke ~size Zeichen: kleine Prompts halten die Antworten kurz."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur)
    return out


def _parse_list(content: str) -> list:
    """Letzte gueltige JSON-Liste in der Antwort (Modelle schreiben Denkprozess davor)."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\[", content))):
        try:
            value, _ = dec.raw_decode(content[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


ANSWER_FORMAT = (
    'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a '
    "VERBATIM excerpt (max 300 chars) copied from the excerpt}. Keep your reasoning "
    "short and end with the JSON list.\n\n"
)
ANSWER_FORMAT_V2 = (   # v5 V3: jede Angabe nennt, fuer welches Modell/welchen Datensatz/welche Einstellung sie gilt, mit Zitat
    'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a '
    'VERBATIM excerpt (max 300 chars) copied from the excerpt, "applies_to": {"model": the model or method this '
    'value belongs to ("all" if the excerpt states it for every model), "dataset": the dataset(s) it belongs to '
    '("all" if general), "setting": "transductive", "inductive" or "not stated"}, "scope_quote": a VERBATIM excerpt '
    "(max 300 chars) that shows for which model and dataset the value holds (it may equal quote)}. If a table, script or "
    "text gives different values for several models or datasets, report each with its own applies_to. "
    "Keep your reasoning short and end with the JSON list.\n\n"
)


def _extract_chunk(llm: LLMClient, chunk: str, claim_text: str, dataset: str, v2: bool = False) -> list:
    resp = llm.complete("extract", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Goal: reproduce this result: {claim_text} (dataset: {dataset}).\n"
            "From the excerpt below (paper prose, README or source code), list every statement or "
            "setting about how the experiment is set up that a reproducer needs: data preprocessing, "
            "data splits, model architecture, initialisation, hyperparameters and their defaults, "
            "optimiser, learning-rate schedule, regularisation, training length and stopping, "
            "evaluation protocol. Include implementation details that change the result: where dropout "
            "is applied (inputs, hidden layers, attention coefficients), feature and adjacency "
            "normalisation, weight initialisation, which parameters get weight decay and how the "
            "regularisation term is scaled, the early-stopping criterion, patience and maximum number "
            "of epochs, and which weights are restored for testing. If the excerpt describes several "
            "protocols (for example the original paper's and a re-evaluation's), say in the topic which "
            "one each value belongs to. Only what this excerpt says; in code, report values exactly as "
            "written. If there is none, answer [].\n"
            + (ANSWER_FORMAT_V2 if v2 else ANSWER_FORMAT) +
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def collect_raw(llm: LLMClient, docs: list[Doc], claim_text: str, dataset: str, v2: bool = False) -> list[dict]:
    """Rohangaben des Modells je Dokument, noch ungeprueft: [{"doc": Name, "item": Objekt}]."""
    jobs = [(doc, chunk) for doc in docs for chunk in _chunks(doc.text[:MAX_TEXT_CHARS])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_lists = list(pool.map(lambda j: _extract_chunk(llm, j[1], claim_text, dataset, v2), jobs))
    return [{"doc": doc.name, "item": item} for (doc, _), lst in zip(jobs, raw_lists) for item in lst]


MIN_FRAGMENT = 10   # kuerzestes Zitatstueck (Zeichen) bei Auslassungen, damit Auslassungspunkte nichts Beliebiges durchlassen


def quote_in_text(quote: str, hay: str, norm=_norm, elisions: bool = True) -> bool:
    """Zitat steht (normalisiert) im Text. Auslassungspunkte am Rand werden ignoriert, im Inneren zerlegen sie das Zitat
    in Stuecke, die der Reihe nach vorkommen muessen (jedes mindestens MIN_FRAGMENT Zeichen). Zahlen und Bezeichner
    bleiben exakt, es wird nur die Schreibweise geglaettet."""
    if not elisions:
        return norm(quote) in hay
    frags = [norm(f) for f in re.split(r"\s*(?:\.\.\.|\u2026)\s*", quote)]
    frags = [f for f in frags if f]
    if not frags or (len(frags) > 1 and any(len(f) < MIN_FRAGMENT for f in frags)):
        return False
    pos = 0
    for f in frags:
        i = hay.find(f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


def numbers_in_quote(value: str, quote: str, norm=_norm) -> bool:
    """Jede Zahl im Wert muss als ganze Zahl im Zitat stehen ('0.005' ist nicht durch '0.05' belegt)."""
    q = norm(quote)
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", q) for n in re.findall(r"\d+(?:\.\d+)?", norm(value)))


def verify_raw(raw: list[dict], docs: list[Doc], norm=_norm, elisions: bool = True, check_numbers: bool = True
               ) -> tuple[list[SetupDetail], list[dict]]:
    """Deterministische Pruefung: Zitat muss (normalisiert) im Dokument stehen, Zahlen des Werts muessen im Zitat stehen.
    Rueckgabe (behalten, verworfen mit Grund)."""
    haystacks = {d.name: norm(d.text) for d in docs}
    kept, dropped, seen = [], [], set()
    for r in raw:
        item, name = r["item"], r["doc"]
        if not isinstance(item, dict):
            dropped.append({**r, "reason": "kein Objekt"})
            continue
        topic, value, quote = (str(item.get(k, "")).strip() for k in ("topic", "value", "quote"))
        if not (topic and value and quote):
            dropped.append({**r, "reason": "Feld fehlt"})
        elif not quote_in_text(quote, haystacks[name], norm, elisions):
            dropped.append({**r, "reason": "Zitat nicht im Quelltext"})
        elif check_numbers and not numbers_in_quote(value, quote, norm):
            dropped.append({**r, "reason": "Zahl des Werts nicht im Zitat"})
        elif (name, norm(quote)) not in seen:
            seen.add((name, norm(quote)))
            kept.append(SetupDetail(topic, value, quote, name))
    return kept, dropped


def extract_setup_details(
    llm: LLMClient, docs: list[Doc], claim_text: str, dataset: str
) -> tuple[list[SetupDetail], int]:
    """Gibt (verifizierte Details, Anzahl verworfener Angaben) zurueck. Zitat muss im Dokument stehen."""
    kept, dropped = verify_raw(collect_raw(llm, docs, claim_text, dataset), docs)
    return kept, len(dropped)


REPO_DEFAULT_MARK = "repository default (the paper may tune this per dataset)"
_DEFAULT_ARG = re.compile(r"default\s*=")


def is_repo_default(d: SetupDetail) -> bool:
    """Wert stammt aus einem argparse-Default (`add_argument(..., default=...)`) einer Repo-Datei, nicht aus dem Paper-Text."""
    return d.source.startswith("repo:") and bool(_DEFAULT_ARG.search(d.quote))


def format_for_prompt(details: list[SetupDetail], mark_defaults: bool = False) -> str:
    """mark_defaults (nur Produktpfad): Repo-Defaults als solche kennzeichnen (M7)."""
    if not details:
        return ""
    lines = [f"- {d.topic}: {d.value}  ({d.source}: \"{d.quote}\")"
             + (f"  [{REPO_DEFAULT_MARK}]" if mark_defaults and is_repo_default(d) else "") for d in details]
    return "Setup details from the paper and its code repository (verified quotes):\n" + "\n".join(lines)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _same_topic(a: str, b: str) -> bool:
    ka, kb = _key(a), _key(b)
    return bool(ka and kb) and (ka == kb or (min(len(ka), len(kb)) >= 4 and (ka in kb or kb in ka)))


def _numbers(value: str) -> tuple[str, ...]:
    return tuple(sorted(re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", value.lower())))


def find_conflicts(details: list[SetupDetail]) -> list[tuple[SetupDetail, SetupDetail]]:
    """Paare (Paper-Angabe, Repo-Angabe) zum selben Thema mit verschiedenem Wert (Zahlen verschieden, sonst Text verschieden)."""
    papers = [d for d in details if d.source == "paper"]
    repos = [d for d in details if d.source.startswith("repo")]
    out = []
    for p in papers:
        for r in repos:
            if not _same_topic(p.topic, r.topic):
                continue
            np_, nr = _numbers(p.value), _numbers(r.value)
            differ = np_ != nr if (np_ and nr) else _key(p.value) != _key(r.value)
            if differ:
                out.append((p, r))
    return out


def format_conflicts(details: list[SetupDetail]) -> str:
    """Plan-Prompt (nur Produktpfad): gleiche Einstellung, verschiedene Werte in Paper und Repo, beide mit Quelle."""
    pairs = find_conflicts(details)
    if not pairs:
        return ""
    lines = [f"- {p.topic}: paper says {p.value} ({p.source}: \"{p.quote}\"); repository says {r.value} ({r.source}: \"{r.quote}\")"
             for p, r in pairs]
    return ("Paper and repository give different values for the same setting. In your plan name both with their source, "
            "and say which one you use and why:\n" + "\n".join(lines))


# ---- v5 Schritt A (Schalter setup_v2): Pflichtliste je Task-Familie (V1), Quellenrang (V2), Feld "gilt fuer" (V3) -----------------
# Die Pflichtliste nennt nur Themen, nie Werte oder Methodenwissen.
RANKS = ("start script", "config", "argparse default", "code", "readme text", "paper")   # hoechster Rang zuerst
OPTIONAL_RE = re.compile(r"\b(optional(ly)?|for example|for instance|e\.g\.|can|may|could|you might)\b", re.I)
DEFAULT_RE = re.compile(r"default\s*=|DEFINE_\w+\(", re.I)
START_RE = re.compile(r"\.sh$|(^|/)(execute|run|train|main|exp)[\w.-]*\.py$", re.I)
CONFIG_RE = re.compile(r"\.(ya?ml|json|conf|cfg|ini|toml)$", re.I)
LOADER_RE = re.compile(r"(^|/)(utils|data|dataset|datasets|load\w*|process\w*|preprocess\w*|prepare\w*|loader\w*)[\w.-]*\.py$|data_loader|data_provider", re.I)
ARGS_RE = re.compile(r"(^|/)(args|config|parser|options)\w*\.py$", re.I)
MAX_TARGET_DOCS = 8

# (Thema, Muster auf das Thema der Angabe, Ausschluss-Muster). Reihenfolge = Reihenfolge im Dossier.
MANDATORY: dict[str, list[tuple[str, str, str]]] = {
    "graph": [
        ("feature normalisation", r"(feature|attribute|input|row|preprocess|standardi)", r"adjacen|laplacian|propagat|graph normali"),
        ("adjacency normalisation", r"adjacen|laplacian|propagat|renormali|symmetric|graph normali", ""),
        ("learning rate", r"learning.?rate|\blr\b", "schedule|decay"),
        ("weight decay", r"weight.?decay|\bl2\b|regulari[sz]", ""),
        ("dropout", r"dropout", ""),
        ("hidden size / heads", r"hidden|head|units|width|channels", ""),
        ("epochs", r"epoch|iteration", ""),
        ("early stopping rule", r"early|patience|stopping", ""),
    ],
    "forecast": [
        ("input length", r"input|seq.?len|look.?back|window|history", "label|pred|horizon"),
        ("learning rate", r"learning.?rate|\blr\b", "schedule|decay|adjust"),
        ("learning-rate schedule", r"schedule|lradj|decay|halv|adjust|anneal", ""),
        ("batch size", r"batch", ""),
        ("epochs", r"epoch|iteration", ""),
        ("patience", r"patience|early|stopping", ""),
    ],
}
GRAPH_DATASETS = ("cora", "citeseer", "pubmed")
GENERIC_MODEL_WORDS = {"model", "models", "all", "our", "ours", "proposed", "baseline", "baselines", "gnn", "gnns", "network", "networks",
                       "neural", "method", "methods", "we", "the", "and", "not", "stated", "any", "every", "general", "default"}
GENERIC_SCOPE = {"", "all", "any", "every", "general", "not stated", "unknown", "n/a", "none", "all datasets", "all models"}


@dataclass(frozen=True)
class SetupDetailV2(SetupDetail):
    applies_to: dict | None = None   # {"model", "dataset", "setting", "quote"}; None = Geltungsbereich nicht belegt
    source_rank: str = ""
    optional: bool = False           # README-Satz mit "optional", "for example", "can" ...
    excluded: str = ""               # Grund, warum die Angabe nicht in den Plan geht (anderes Modell/Datensatz/Einstellung)


def family_for(dataset: str) -> str:
    return "graph" if any(g in dataset.lower() for g in GRAPH_DATASETS) else "forecast"


def source_rank(doc_name: str, quote: str) -> str:
    if doc_name == "paper":
        return "paper"
    if doc_name == "repo-readme":
        return "readme text"
    path = doc_name.split(":", 1)[-1].split("@")[0]
    if DEFAULT_RE.search(quote):
        return "argparse default"
    if START_RE.search(path):
        return "start script"
    if CONFIG_RE.search(path):
        return "config"
    return "code"


def _scope(item: dict, doc_norm: str, norm=_norm) -> dict | None:
    ap, sq = item.get("applies_to"), str(item.get("scope_quote", "")).strip()
    if not isinstance(ap, dict) or not sq or not quote_in_text(sq, doc_norm, norm):
        return None   # Geltungsbereich ohne woertliches Zitat = nicht belegt
    return {"model": str(ap.get("model", "")).strip(), "dataset": str(ap.get("dataset", "")).strip(),
            "setting": str(ap.get("setting", "")).strip().lower(), "quote": sq}


def verify_raw_v2(raw: list[dict], docs: list[Doc], norm=_norm) -> tuple[list[SetupDetailV2], list[dict]]:
    """Wie verify_raw (Zitat und Zahlen), dazu Quellenrang, Markierung optional und Geltungsbereich mit eigenem Zitat."""
    kept, dropped = verify_raw(raw, docs, norm)
    keep_keys = {(d.source, norm(d.quote)) for d in kept}
    hay = {d.name: norm(d.text) for d in docs}
    by_key = {}
    for r in raw:
        item = r["item"]
        if isinstance(item, dict):
            by_key.setdefault((r["doc"], norm(str(item.get("quote", "")).strip())), item)
    out = []
    for d in kept:
        item = by_key.get((d.source, norm(d.quote)), {})
        rank = source_rank(d.source, d.quote)
        out.append(SetupDetailV2(d.topic, d.value, d.quote, d.source, _scope(item, hay[d.source], norm), rank,
                                 rank == "readme text" and bool(OPTIONAL_RE.search(d.quote))))
    return out, dropped


def scope_exclusion(d: SetupDetailV2, claim_text: str, dataset: str) -> str:
    """Grund, warum die Angabe nicht fuer diesen Claim gilt ("" = gilt oder Geltungsbereich unbekannt). Nur belegte Bereiche zaehlen."""
    ap = d.applies_to
    if not ap:
        return ""
    claim_n, data_n = _norm(claim_text), _norm(dataset)
    model = _norm(ap.get("model", ""))
    tokens = [t for t in re.split(r"[^a-z0-9]+", model) if len(t) >= 3 and t not in GENERIC_MODEL_WORDS]
    if tokens and not any(t in claim_n for t in tokens):
        return f"other model ({ap['model']})"
    ds = _norm(ap.get("dataset", ""))
    ds_tokens = [t for t in re.split(r"[^a-z0-9]+", ds) if len(t) >= 3 and t not in GENERIC_MODEL_WORDS]
    if ds not in GENERIC_SCOPE and ds_tokens and not any(t in data_n or any(w in t for w in data_n.split() if len(w) >= 3) for t in ds_tokens):
        return f"other dataset ({ap['dataset']})"
    setting = ap.get("setting", "")
    if family_for(dataset) == "graph" and "inductive" in setting and "transductive" not in setting:
        return "inductive setting"
    return ""


def apply_scope(details: list[SetupDetailV2], claim_text: str, dataset: str) -> list[SetupDetailV2]:
    return [dataclasses.replace(d, excluded=scope_exclusion(d, claim_text, dataset)) if not d.excluded else d for d in details]


def missing_topics(details: list[SetupDetailV2], family: str) -> list[str]:
    """Pflichtthemen ohne gueltige (nicht ausgeschlossene) Angabe; nur das Thema der Angabe zaehlt, nie ihr Wert."""
    live = [d.topic for d in details if not d.excluded]
    out = []
    for name, inc, exc in MANDATORY[family]:
        if not any(re.search(inc, t, re.I) and not (exc and re.search(exc, t, re.I)) for t in live):
            out.append(name)
    return out


def target_docs(docs: list[Doc]) -> list[Doc]:
    """Dateien, die Daten laden, und Startskripte/Argumentdefinitionen des Experiments, plus README (deterministisch nach Pfad)."""
    def path(d: Doc) -> str:
        return d.name.split(":", 1)[-1].split("@")[0]
    repo = [d for d in docs if d.name.startswith("repo:")]
    picked = [d for d in repo if START_RE.search(path(d)) or ARGS_RE.search(path(d)) or CONFIG_RE.search(path(d))]
    picked += [d for d in repo if LOADER_RE.search(path(d)) and d not in picked]
    picked += [d for d in docs if d.name == "repo-readme"]
    return picked[:MAX_TARGET_DOCS]


def _targeted_chunk(llm: LLMClient, chunk: str, claim_text: str, dataset: str, topics: list[str]) -> list:
    resp = llm.complete("extract", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Goal: reproduce this result: {claim_text} (dataset: {dataset}).\n"
            "Earlier passes found no statement for these settings: " + "; ".join(topics) + ".\n"
            "From the excerpt below (code, script, config or README), list only statements that set or describe exactly these "
            "settings for the experiment of the goal, with the value exactly as written. If the excerpt has none, answer [].\n"
            + ANSWER_FORMAT_V2 + f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def extract_setup_details_v2(
    llm: LLMClient, docs: list[Doc], claim_text: str, dataset: str
) -> tuple[list[SetupDetailV2], int, list[str], dict]:
    """V1 bis V3. Rueckgabe: (Angaben inkl. ausgeschlossener, Zahl verworfener, nicht gefundene Pflichtthemen, Info).
    Ablauf: erste Extraktion, Geltungsbereich pruefen, fehlende Pflichtthemen gezielt in Datenlade- und Startdateien suchen,
    erneut pruefen. Jede Angabe braucht ein woertliches Zitat (verify_raw), "nicht gefunden" ist ein gueltiges Ergebnis."""
    family = family_for(dataset)
    details, dropped = verify_raw_v2(collect_raw(llm, docs, claim_text, dataset, v2=True), docs)
    details = apply_scope(details, claim_text, dataset)
    missing = missing_topics(details, family)
    info = {"family": family, "missing_first": list(missing), "targeted_docs": [], "added": 0}
    n_dropped = len(dropped)
    if missing:
        targets = target_docs(docs)
        info["targeted_docs"] = [d.name for d in targets]
        jobs = [(d, c) for d in targets for c in _chunks(d.text[:MAX_TEXT_CHARS])]
        if jobs:
            with ThreadPoolExecutor(max_workers=4) as pool:
                lists = list(pool.map(lambda j: _targeted_chunk(llm, j[1], claim_text, dataset, missing), jobs))
            raw2 = [{"doc": d.name, "item": it} for (d, _), lst in zip(jobs, lists) for it in lst]
            extra, dropped2 = verify_raw_v2(raw2, targets)
            seen = {(d.source, _norm(d.quote)) for d in details}
            new = [d for d in apply_scope(extra, claim_text, dataset) if (d.source, _norm(d.quote)) not in seen]
            details += new
            info["added"] = len(new)
            n_dropped += len(dropped2)
        missing = missing_topics(details, family)
    return details, n_dropped, missing, info


def _rank_index(d: SetupDetailV2) -> int:
    return RANKS.index(d.source_rank) if d.source_rank in RANKS else len(RANKS)


def plan_details_v2(details: list[SetupDetail]) -> list[SetupDetailV2]:
    return [d for d in details if isinstance(d, SetupDetailV2) and not d.excluded]


def find_conflicts_v2(details: list[SetupDetailV2]) -> list[list[SetupDetailV2]]:
    """Gruppen (nach Rang sortiert) von Angaben zum selben Thema mit verschiedenem Wert (nur Angaben, die in den Plan gehen)."""
    live = sorted(plan_details_v2(details), key=_rank_index)
    groups: list[list[SetupDetailV2]] = []
    for d in live:
        for g in groups:
            if _same_topic(g[0].topic, d.topic):
                g.append(d)
                break
        else:
            groups.append([d])
    out = []
    for g in groups:
        first = g[0]
        diff = [x for x in g[1:] if (_numbers(first.value) != _numbers(x.value) if (_numbers(first.value) and _numbers(x.value))
                                      else _key(first.value) != _key(x.value))]
        if diff:
            out.append([first] + diff)
    return out


def format_for_prompt_v2(details: list[SetupDetail]) -> str:
    live = plan_details_v2(details)
    if not live:
        return ""
    lines = []
    for d in live:
        marks = (f"  [optional advice, not part of the experiment]" if d.optional else "") + \
                (f"  [{REPO_DEFAULT_MARK}]" if d.source_rank == "argparse default" else "")
        lines.append(f"- {d.topic}: {d.value}  ({d.source} [{d.source_rank}]: \"{d.quote}\"){marks}")
    return ("Setup details from the paper and its code repository (verified quotes). Source rank, highest first: "
            + ", ".join(RANKS) + ".\n" + "\n".join(lines))


def format_conflicts_v2(details: list[SetupDetail]) -> str:
    groups = find_conflicts_v2(details)
    if not groups:
        return ""
    lines = []
    for g in groups:
        lines.append(f"- {g[0].topic}: " + "; ".join(f"{x.value} ({x.source} [{x.source_rank}])" for x in g)
                     + f" -> use the value of the highest-ranked source ({g[0].source_rank})")
    return ("Sources give different values for the same setting. In your plan name both values with their source and use the value of "
            "the highest-ranked source:\n" + "\n".join(lines))

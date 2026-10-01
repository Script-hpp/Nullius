"""Tavily-Rolle 4: Setup-Details (Vorverarbeitung, Split, Hyperparameter, Training) aus dem Paper.

Allgemein gehalten: fragt nach ALLEN Angaben zum Versuchsaufbau, nicht nach einem bestimmten
Detail. Wie bei Rolle 2 gilt: die KI liest, ein Programm prueft. Jede Angabe muss mit einem
woertlichen Zitat belegt sein, das im Quelltext steht; sonst wird sie verworfen.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
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


def _extract_chunk(llm: LLMClient, chunk: str, claim_text: str, dataset: str) -> list:
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
            'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a '
            "VERBATIM excerpt (max 300 chars) copied from the excerpt}. Keep your reasoning "
            "short and end with the JSON list.\n\n"
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def collect_raw(llm: LLMClient, docs: list[Doc], claim_text: str, dataset: str) -> list[dict]:
    """Rohangaben des Modells je Dokument, noch ungeprueft: [{"doc": Name, "item": Objekt}]."""
    jobs = [(doc, chunk) for doc in docs for chunk in _chunks(doc.text[:MAX_TEXT_CHARS])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_lists = list(pool.map(lambda j: _extract_chunk(llm, j[1], claim_text, dataset), jobs))
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


def format_for_prompt(details: list[SetupDetail]) -> str:
    if not details:
        return ""
    lines = [f"- {d.topic}: {d.value}  ({d.source}: \"{d.quote}\")" for d in details]
    return "Setup details from the paper and its code repository (verified quotes):\n" + "\n".join(lines)

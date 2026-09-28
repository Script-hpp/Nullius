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
            "evaluation protocol. Only what this excerpt says; in code, report values exactly as "
            "written. If there is none, answer [].\n"
            'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a '
            "VERBATIM excerpt (max 300 chars) copied from the excerpt}. Keep your reasoning "
            "short and end with the JSON list.\n\n"
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def extract_setup_details(
    llm: LLMClient, docs: list[Doc], claim_text: str, dataset: str
) -> tuple[list[SetupDetail], int]:
    """Gibt (verifizierte Details, Anzahl verworfener Angaben) zurueck. Zitat muss im Dokument stehen."""
    jobs = [(doc, chunk) for doc in docs for chunk in _chunks(doc.text[:MAX_TEXT_CHARS])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_lists = list(pool.map(lambda j: _extract_chunk(llm, j[1], claim_text, dataset), jobs))
    haystacks = {d.name: _norm(d.text) for d in docs}
    kept, dropped, seen = [], 0, set()
    for (doc, _), lst in zip(jobs, raw_lists):
        for item in lst:
            if not isinstance(item, dict):
                dropped += 1
                continue
            topic, value, quote = (str(item.get(k, "")).strip() for k in ("topic", "value", "quote"))
            nq = _norm(quote)
            if not (topic and value and quote and nq in haystacks[doc.name]):
                dropped += 1
            elif (doc.name, nq) not in seen:
                seen.add((doc.name, nq))
                kept.append(SetupDetail(topic, value, quote, doc.name))
    return kept, dropped


def format_for_prompt(details: list[SetupDetail]) -> str:
    if not details:
        return ""
    lines = [f"- {d.topic}: {d.value}  ({d.source}: \"{d.quote}\")" for d in details]
    return "Setup details from the paper and its code repository (verified quotes):\n" + "\n".join(lines)

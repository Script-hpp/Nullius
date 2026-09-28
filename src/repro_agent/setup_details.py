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

from .claims_source import PaperText, _norm
from .llm import LLMClient

MAX_TEXT_CHARS = 60_000
CHUNK_CHARS = 5_000

SYSTEM = (
    "You extract experimental-setup facts from a research paper so someone else can reproduce a "
    "result. Keep reasoning brief."
)


@dataclass(frozen=True)
class SetupDetail:
    topic: str
    value: str
    quote: str


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
    resp = llm.complete("reasoning", [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Goal: reproduce this result: {claim_text} (dataset: {dataset}).\n"
            "From the text excerpt below, list every statement about how the experiment is set "
            "up that a reproducer needs: data preprocessing, data splits, model architecture, "
            "initialisation, hyperparameters, optimiser, regularisation, training length and "
            "stopping, evaluation protocol. Only what this excerpt says. If there is none, "
            "answer [].\n"
            'Answer with a JSON list of {"topic": short label, "value": the setting, "quote": a '
            "VERBATIM excerpt (max 300 chars) copied from the excerpt}. Keep your reasoning "
            "short and end with the JSON list.\n\n"
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def extract_setup_details(
    llm: LLMClient, paper: PaperText, claim_text: str, dataset: str
) -> tuple[list[SetupDetail], int]:
    """Gibt (verifizierte Details, Anzahl verworfener Angaben) zurueck."""
    chunks = _chunks(paper.text[:MAX_TEXT_CHARS])
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_lists = list(pool.map(lambda c: _extract_chunk(llm, c, claim_text, dataset), chunks))
    haystack = _norm(paper.text)
    kept, dropped, seen = [], 0, set()
    for item in (x for lst in raw_lists for x in lst):
        if not isinstance(item, dict):
            dropped += 1
            continue
        topic, value, quote = (str(item.get(k, "")).strip() for k in ("topic", "value", "quote"))
        if not (topic and value and quote and _norm(quote) in haystack):
            dropped += 1
        elif _norm(quote) not in seen:
            seen.add(_norm(quote))
            kept.append(SetupDetail(topic, value, quote))
    return kept, dropped


def format_for_prompt(details: list[SetupDetail]) -> str:
    if not details:
        return ""
    lines = [f"- {d.topic}: {d.value}  (paper: \"{d.quote}\")" for d in details]
    return "Setup details stated in the paper (verified quotes):\n" + "\n".join(lines)

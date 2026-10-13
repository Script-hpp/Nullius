"""Tavily-Rolle 3: Einwaende aus gesammelten Seiten (Issues, Kritik) lesen und pruefen.

Jede Aussage braucht ein woertliches Zitat, das im Quelltext steht; der Quelltext muss das Paper nennen
(Stichwoerter). Zusaetzlich deterministischer Abgleich mit dem Repo (Beispiel: `drop_last` beim Test).
"""

from __future__ import annotations

from .spend_guard import context_map

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .claims_source import _norm
from .llm import LLMClient
from .setup_details import Doc, _chunks, _parse_list

PROTOCOL_KEYWORDS = ("drop_last", "last batch", "test set", "leakage", "hyperparameter", "validation set", "test data")


@dataclass
class Objection:
    topic: str
    statement: str
    quote: str
    source: str            # URL der Seite
    protocol: bool         # nennt ein Stichwort zum Auswertungsprotokoll
    repo_check: str        # bestaetigt | widerlegt | nicht pruefbar


def docs_from_collect(collected: dict, must_mention: tuple[str, ...], exempt_prefixes: tuple[str, ...] = ()) -> list[Doc]:
    """Seiten aus Search (raw_content) und Crawl; nur solche, die das Paper nennen (Stichwoerter, klein geschrieben).
    Seiten, deren URL mit einem Praefix aus `exempt_prefixes` beginnt (klein geschrieben; Issues des Paper-Repos), brauchen den Namen nicht:
    ein Issue im Repo des Papers spricht vom Repo, nicht vom Titel."""
    docs, seen = [], set()
    items = list((collected.get("search") or {}).get("results", [])) + list((collected.get("crawl") or {}).get("results", []))
    for it in items:
        url, text = it.get("url", ""), it.get("raw_content") or it.get("content") or ""
        if not text or url in seen:
            continue
        seen.add(url)
        exempt = bool(exempt_prefixes) and url.lower().startswith(tuple(exempt_prefixes))
        if exempt or any(k.lower() in text.lower() for k in must_mention):
            docs.append(Doc(url, text))
    return docs


def _extract_chunk(llm: LLMClient, chunk: str, paper: str, dataset: str) -> list:
    resp = llm.complete("extract", [
        {"role": "system", "content": "You extract objections from web pages. Answer only from the text you are given."},
        {"role": "user", "content": (
            f"Paper: {paper}. Dataset: {dataset}.\n"
            "From the excerpt, list every statement that criticises or questions how the paper's results were "
            "evaluated or whether they can be reproduced (evaluation protocol, test set handling, dropped last "
            "batch, data leakage, hyperparameter search on test data, results that could not be reproduced). "
            "Only what the excerpt says. If there is none, answer [].\n"
            'Answer with a JSON list of {"topic": short label, "statement": the objection in one sentence, '
            '"quote": a VERBATIM excerpt (max 300 chars) copied from the excerpt}. Keep your reasoning short '
            "and end with the JSON list.\n\n"
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def extract_objections(
    llm: LLMClient, docs: list[Doc], paper: str, dataset: str, repo_files: dict[str, str] | None = None,
    drops: list[dict] | None = None, candidates: list[dict] | None = None,
) -> tuple[list[Objection], int]:
    """`drops` (optional) sammelt je verworfenem Kandidaten {url, reason, topic, statement, quote}; `candidates` (optional) alle
    rohen Extraktor-Kandidaten je Seite {url, topic, statement, quote, quote_in_page}, auch die behaltenen."""
    jobs = [(d, c) for d in docs for c in _chunks(d.text[:60000])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = list(context_map(pool, lambda j: _extract_chunk(llm, j[1], paper, dataset), jobs))
    hay = {d.name: _norm(d.text) for d in docs}
    kept, dropped, seen = [], 0, set()
    for (doc, _), lst in zip(jobs, raw):
        for it in lst:
            if not isinstance(it, dict):
                dropped += 1
                if drops is not None:
                    drops.append({"url": doc.name, "reason": "not_an_object", "topic": "", "statement": "", "quote": str(it)[:200]})
                continue
            topic, stmt, quote = (str(it.get(k, "")).strip() for k in ("topic", "statement", "quote"))
            nq = _norm(quote)
            if candidates is not None:
                candidates.append({"url": doc.name, "topic": topic, "statement": stmt[:300], "quote": quote[:300],
                                   "quote_in_page": bool(stmt and quote and nq in hay[doc.name])})
            if not (stmt and quote and nq in hay[doc.name]):
                dropped += 1
                if drops is not None:
                    reason = "empty_statement_or_quote" if not (stmt and quote) else "quote_not_in_page"
                    drops.append({"url": doc.name, "reason": reason, "topic": topic, "statement": stmt[:300], "quote": quote[:300]})
            elif (doc.name, nq) not in seen:
                seen.add((doc.name, nq))
                low = (stmt + " " + quote).lower()
                kept.append(Objection(topic, stmt, quote, doc.name,
                                      any(k in low for k in PROTOCOL_KEYWORDS), repo_check(stmt + " " + quote, repo_files)))
    return kept, dropped


def repo_check(text: str, repo_files: dict[str, str] | None) -> str:
    """Deterministischer Abgleich: behauptet der Einwand einen `drop_last`-Wert beim Test, steht im Repo (data_factory.py) das Gegenteil?"""
    if not repo_files or "drop_last" not in text.lower():
        return "nicht pruefbar"
    src = next((t for p, t in repo_files.items() if p.endswith("data_factory.py")), None)
    if src is None:
        return "nicht pruefbar"
    m = re.search(r"flag\s*==\s*['\"]test['\"].*?drop_last\s*=\s*(True|False)", src, flags=re.S)
    if not m:
        return "nicht pruefbar"
    repo_true = m.group(1) == "True"
    claims_true = bool(re.search(r"drop_last\W{0,6}(=|is|to|set to)?\W{0,6}true|drops? the last", text, re.I))
    claims_false = bool(re.search(r"drop_last\W{0,6}(=|is|to|set to)?\W{0,6}false", text, re.I))
    if claims_true == claims_false:
        return "nicht pruefbar"
    return "bestaetigt" if claims_true == repo_true else "widerlegt"

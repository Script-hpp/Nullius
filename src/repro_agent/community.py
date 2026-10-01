"""Community-Check nach dem Urteil: "What others found" (Rolle 3) und "Possible explanations" (Rolle 6).

Grundsatz: nur berichten, nie das Urteil aendern. Der Check laeuft erst, wenn alle Urteile feststehen; er schreibt nur in eigene
Dossier-Felder und geht nie in Plan, Code oder Urteil. Jede Aussage braucht ein woertliches Zitat, das im Quelltext steht; der Quelltext muss
das Paper nennen (arXiv-ID oder voller Titel). Die Etiketten sind deterministisch (Stichwoerter), keine LLM-Einstufung "bestaetigt/widerspricht".

Kontingent: Ergebnis je arXiv-ID in `logs/community_cache.json` (7 Tage), Discovery-Community-Treffer des Katalogs ersetzen die Suche
(ein Extract statt einer Search), Crawl der Repo-Issues nur, wenn das Repo bekannt ist und die Suche keine Issue-Seite dieses Repos lieferte.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .claims_source import _norm
from .dossier import REPO_ROOT
from .objections import Objection, docs_from_collect, extract_objections
from .setup_details import Doc, _chunks, _parse_list

CACHE_PATH = REPO_ROOT / "logs" / "community_cache.json"
CACHE_TTL_S = 7 * 86400
MAX_PAGES = 5            # Treffer der Suche
MAX_CRAWL = 3            # Seiten des Issue-Crawls (immer limit)
MAX_READ = 3             # Discovery-Treffer, die statt einer Suche gelesen werden
MAX_CHUNKS = 12          # extract-Aufrufe pro Check (Kosten)
DEVIATING = ("teilweise", "nicht reproduziert")
KIND_PROTOCOL, KIND_REPORT, KIND_OTHER = "protocol note", "reproduction report", "other"
REPORT_RE = re.compile(r"(could not|couldn.t|cannot|can.t|unable to|fail(?:ed|s)? to|not able to)\W+(?:\w+\W+){0,3}?reproduc|"
                       r"not reproducible|reproduction|replicat|reproduc", re.I)


@dataclass
class Finding:
    topic: str
    statement: str
    quote: str
    url: str
    kind: str
    repo_check: str


def classify(statement: str, quote: str, protocol: bool) -> str:
    """Deterministisch: Protokollhinweis (Stichworte aus objections.PROTOCOL_KEYWORDS) vor Reproduktionsbericht vor sonstiges."""
    if protocol:
        return KIND_PROTOCOL
    return KIND_REPORT if REPORT_RE.search(f"{statement} {quote}") else KIND_OTHER


def keywords(title: str, arxiv_id: str) -> tuple[str, ...]:
    """Eine Seite gehoert zum Paper, wenn sie die arXiv-ID oder den vollen Titel nennt."""
    return tuple(k for k in (arxiv_id.lower(), " ".join((title or "").lower().split())) if k)


def _issue_hit_of(repo_url: str | None, urls: list[str]) -> bool:
    if not repo_url:
        return False
    base = repo_url.rstrip("/").lower().split("github.com/")[-1]
    return any(u.lower().split("github.com/")[-1].startswith(base + "/issues") for u in urls if "github.com/" in u.lower())


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))


def cache_get(arxiv_id: str, now: float | None = None, path: Path | None = None) -> dict | None:
    entry = _load_cache(path or CACHE_PATH).get(arxiv_id)
    if not entry or (now or time.time()) - entry.get("ts", 0) > CACHE_TTL_S:
        return None
    return entry


def cache_put(arxiv_id: str, entry: dict, path: Path | None = None) -> None:
    p = path or CACHE_PATH
    cache = _load_cache(p)
    cache[arxiv_id] = entry
    _save_cache(p, cache)


def _pages(docs: list[Doc]) -> list[dict]:
    import hashlib
    return [{"url": d.name, "n_chars": len(d.text), "sha256": hashlib.sha256(d.text.encode()).hexdigest()} for d in docs]


def what_others_found(tools, llm, title: str, arxiv_id: str, dataset: str, repo_url: str | None,
                      repo_files: dict[str, str] | None, prior_hits: list[dict] | None, now: float | None = None) -> dict:
    """Ergebnis: {"findings": [...], "dropped", "pages": [...], "queries": [...], "reused_discovery": bool, "calls": [...]}.
    Aufrufe: Search (oder Extract der Discovery-Treffer) und hoechstens ein Issue-Crawl."""
    calls, queries = [], []
    prior_urls = [h.get("url") for h in (prior_hits or []) if h.get("url")]
    if prior_urls:   # Discovery hat schon Treffer: URLs lesen statt neu zu suchen
        res = tools.read_pages(prior_urls, MAX_READ)
        calls.append("extract")
        queries.append({"kind": "discovery_hits", "urls": prior_urls[:MAX_READ]})
        urls_seen = prior_urls
    else:
        res = tools.search_community(title, MAX_PAGES)
        calls.append("search")
        queries.append({"kind": "search", "title": title})
        urls_seen = [r.get("url", "") for r in (res or {}).get("results", [])]
    crawl = None
    if repo_url and not _issue_hit_of(repo_url, urls_seen):
        crawl = tools.crawl_issues(repo_url, MAX_CRAWL)
        calls.append("crawl")
        queries.append({"kind": "crawl", "url": repo_url.rstrip("/") + "/issues"})
    docs = docs_from_collect({"search": res, "crawl": crawl}, keywords(title, arxiv_id))
    kept, dropped = extract_objections(llm, docs[:MAX_CHUNKS], title, dataset, repo_files)
    findings = [asdict(Finding(o.topic, o.statement, o.quote, o.source, classify(o.statement, o.quote, o.protocol), o.repo_check)) for o in kept]
    return {"findings": findings, "dropped": dropped, "pages": _pages(docs), "queries": queries,
            "reused_discovery": bool(prior_urls), "calls": calls}


EXPLAIN_SYSTEM = "You extract possible explanations from web pages. Answer only from the text you are given."


def _explain_chunk(llm, chunk: str, title: str, dataset: str) -> list:
    resp = llm.complete("extract", [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": (
            f"Paper: {title}. Dataset: {dataset}.\n"
            "Someone re-implemented the paper and measured a different number than the paper reports. From the excerpt, list every "
            "statement that could explain such a difference: errata, a newer or corrected version of the paper, a known evaluation or "
            "data-loading bug, different protocol or split, hyperparameters that the paper does not state, results that others could "
            "not reproduce. Only what the excerpt says. If there is none, answer [].\n"
            'Answer with a JSON list of {"topic": short label, "explanation": one sentence, "quote": a VERBATIM excerpt (max 300 chars) '
            "copied from the excerpt}. Keep your reasoning short and end with the JSON list.\n\n"
            f"EXCERPT:\n{chunk}"
        )},
    ], temperature=0.0, max_tokens=6000, timeout_s=600)
    return _parse_list(resp.content)


def possible_explanations(tools, llm, title: str, arxiv_id: str, dataset: str) -> dict:
    """Rolle 6: eine Search nach Errata/neuerer Version/Issues; jede Erklaerung mit geprueftem Zitat, ungepruefte Hypothese. Keine Branches."""
    res = tools.clarify_discrepancy(title)
    docs = docs_from_collect({"search": res}, keywords(title, arxiv_id))[:MAX_CHUNKS]
    jobs = [(d, c) for d in docs for c in _chunks(d.text[:60000])][:MAX_CHUNKS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = list(pool.map(lambda j: _explain_chunk(llm, j[1], title, dataset), jobs))
    hay = {d.name: _norm(d.text) for d in docs}
    items, dropped, seen = [], 0, set()
    for (doc, _), lst in zip(jobs, raw):
        for it in lst:
            if not isinstance(it, dict):
                dropped += 1
                continue
            topic, expl, quote = (str(it.get(k, "")).strip() for k in ("topic", "explanation", "quote"))
            nq = _norm(quote)
            if not (expl and quote and nq in hay[doc.name]):
                dropped += 1
            elif (doc.name, nq) not in seen:
                seen.add((doc.name, nq))
                items.append({"topic": topic, "explanation": expl, "quote": quote, "url": doc.name, "status": "unverified hypothesis"})
    return {"items": items, "dropped": dropped, "pages": _pages(docs), "calls": ["search"]}

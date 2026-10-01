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
from .objections import Objection, extract_objections
from .setup_details import Doc, _chunks, _parse_list

CACHE_PATH = REPO_ROOT / "logs" / "community_cache.json"
CACHE_TTL_S = 7 * 86400
CACHE_VERSION = 2        # 2: Seitenfilter, Methoden-Query, Fund muss Methode oder Titel nennen (aeltere Eintraege werden ignoriert)
MAX_PAGE_CHARS = 60_000  # laengere Seiten sind Sammlungen/Dumps, keine Diskussion
COLLECTION_RE = re.compile(r"gist\.github\.com|awesome[-_]|paper[-_]?club|reading[-_ ]?list|papers?[-_]we[-_]love|/awesome", re.I)
MAX_PAGES = 5            # Treffer der Suche
MAX_CRAWL = 3            # Seiten des Issue-Crawls (immer limit)
MAX_READ = 3             # Discovery-Treffer, die statt einer Suche gelesen werden
MAX_CHUNKS = 12          # extract-Aufrufe pro Check (Kosten)
DEVIATING = ("teilweise", "nicht reproduziert")
KIND_PROTOCOL, KIND_REPORT, KIND_OTHER = "protocol note", "reproduction report", "other"
REPORT_RE = re.compile(r"(could not|couldn.t|cannot|can.t|unable to|fail(?:ed|s)? to|not able to)\W+(?:\w+\W+){0,3}?reproduc|"
                       r"not reproducible|reproduction|replicat|reproduc", re.I)


KIND_LABEL = {KIND_PROTOCOL: "Protocol note", KIND_REPORT: "Reproduction report", KIND_OTHER: "Note"}


@dataclass
class Finding:
    topic: str
    statement: str
    quote: str
    url: str
    kind: str
    repo_check: str
    label: str = ""          # Anzeigename der Art (aus `kind`), damit kein Leser ein leeres Etikett sieht


def classify(statement: str, quote: str, protocol: bool) -> str:
    """Deterministisch: Protokollhinweis (Stichworte aus objections.PROTOCOL_KEYWORDS) vor Reproduktionsbericht vor sonstiges."""
    if protocol:
        return KIND_PROTOCOL
    return KIND_REPORT if REPORT_RE.search(f"{statement} {quote}") else KIND_OTHER


def keywords(title: str, arxiv_id: str) -> tuple[str, ...]:
    """Eine Seite gehoert zum Paper, wenn sie die arXiv-ID oder den vollen Titel nennt."""
    return tuple(k for k in (arxiv_id.lower(), " ".join((title or "").lower().split())) if k)


NOISE_HOSTS = ("scirp.org", "semanticscholar.org", "researchgate.net", "scholar.google", "paperswithcode.com", "connectedpapers.com")


def repo_slug(repo_url: str | None) -> str | None:
    m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", (repo_url or "").rstrip("/"))
    return f"{m.group(1)}/{m.group(2)}" if m else None


def drop_noise(results: dict | None, arxiv_id: str) -> dict:
    """Entfernt die eigene arXiv-Seite des Papers (abs/pdf/html, jede Version) und Zitier-Aggregatoren aus Treffern."""
    own = re.compile(rf"arxiv\.org/(?:abs|pdf|html)/{re.escape(arxiv_id)}(?:v\d+)?(?:\.pdf)?/?$", re.I)
    keep = [r for r in (results or {}).get("results", [])
            if not own.search(r.get("url", "")) and not any(h in r.get("url", "").lower() for h in NOISE_HOSTS)]
    return {**(results or {}), "results": keep}


def _issue_hit_of(repo_url: str | None, urls: list[str]) -> bool:
    slug = repo_slug(repo_url)
    return bool(slug) and any(u.lower().startswith(tuple(f"https://github.com/{slug.lower()}/{k}" for k in ("issues", "pull", "discussions")))
                              for u in urls)


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
    if not entry or entry.get("v") != CACHE_VERSION or (now or time.time()) - entry.get("ts", 0) > CACHE_TTL_S:
        return None
    return entry


def cache_put(arxiv_id: str, entry: dict, path: Path | None = None) -> None:
    p = path or CACHE_PATH
    cache = _load_cache(p)
    cache[arxiv_id] = entry
    _save_cache(p, cache)


def is_collection(url: str) -> bool:
    """Sammelseiten (gists, awesome-Listen, Paper-Clubs, Leselisten) sind keine Diskussion eines Papers."""
    return bool(COLLECTION_RE.search(url or ""))


def _word(text: str, name: str) -> bool:
    return bool(name) and re.search(rf"(?<![A-Za-z0-9]){re.escape(name.lower())}(?![A-Za-z0-9])", text.lower()) is not None


def names_paper(text: str, title: str, arxiv_id: str, method: str | None) -> bool:
    """Der Text nennt den Methodennamen (ganzes Wort), den vollen Titel oder die arXiv-ID."""
    low = " ".join(text.lower().split())
    return (" ".join((title or "").lower().split()) in low and bool(title)) or (bool(arxiv_id) and arxiv_id.lower() in low) or _word(text, method or "")


def select_docs(items: list[dict], title: str, arxiv_id: str, exempt: tuple[str, ...] = (), method: str | None = None,
                dataset: str | None = None, method_hits: set[str] = frozenset()) -> list[Doc]:
    """Seiten, die gelesen werden: kein Duplikat, nicht leer, <= MAX_PAGE_CHARS, keine Sammelseite. Seiten des Paper-Repos (exempt) sind
    gesetzt; sonst muss der Text das Paper nennen (arXiv-ID oder voller Titel); Treffer der Methoden-Query (`method_hits`, URLs) auch mit
    Methodenname plus Datensatz im Text (Kritik-Papers nennen selten den Titel)."""
    docs, seen = [], set()
    kws = keywords(title, arxiv_id)
    for it in items:
        url, text = it.get("url", ""), it.get("raw_content") or it.get("content") or ""
        if not text or url in seen or len(text) > MAX_PAGE_CHARS or is_collection(url):
            continue
        seen.add(url)
        low = text.lower()
        if exempt and url.lower().startswith(tuple(exempt)):
            docs.append(Doc(url, text))
        elif any(k in low for k in kws) or (url in method_hits and _word(text, method or "") and _word(text, dataset or "")):
            docs.append(Doc(url, text))
    return docs


def _pages(docs: list[Doc]) -> list[dict]:
    import hashlib
    return [{"url": d.name, "n_chars": len(d.text), "sha256": hashlib.sha256(d.text.encode()).hexdigest()} for d in docs]


def what_others_found(tools, llm, title: str, arxiv_id: str, dataset: str, repo_url: str | None,
                      repo_files: dict[str, str] | None, prior_hits: list[dict] | None, now: float | None = None,
                      method: str | None = None) -> dict:
    """Ergebnis: {"findings": [...], "dropped", "pages": [...], "queries": [...], "reused_discovery": bool, "calls": [...]}.
    Aufrufe: Search (oder Extract der Discovery-Treffer); bei bekanntem Repo ohne Issue-Treffer aus ihm eine Search auf dessen Issues
    und nur wenn auch die nichts liefert ein Issue-Crawl. Zitier-Aggregatoren und die eigene arXiv-Seite fallen heraus, Seiten des
    Paper-Repos brauchen den Titel nicht. Mit bekannter Methode kommt eine methodenzentrierte Suche auf arxiv.org/openreview.net dazu
    (Kritik-Papers wie Shchur 2018). Seiten > 60k Zeichen und Sammelseiten fallen heraus; ein Fund bleibt nur, wenn Zitat oder Satz Methode,
    Titel oder arXiv-ID nennen (Repo-Seiten ausgenommen)."""
    calls, queries = [], []
    prior_urls = [h.get("url") for h in (prior_hits or []) if h.get("url")]
    if prior_urls:   # Discovery hat schon Treffer: URLs lesen statt neu zu suchen
        res = tools.read_pages(prior_urls, MAX_READ)
        calls.append("extract")
        queries.append({"kind": "discovery_hits", "urls": prior_urls[:MAX_READ]})
    else:
        res = tools.search_community(title, MAX_PAGES)
        calls.append("search")
        queries.append({"kind": "search", "title": title})
    res = drop_noise(res, arxiv_id)
    method_res = None
    if method and dataset:
        method_res = drop_noise(tools.search_method_critique(method, dataset, MAX_PAGES), arxiv_id)
        calls.append("search")
        queries.append({"kind": "method_search", "method": method, "dataset": dataset})
    urls_seen = [r.get("url", "") for r in res.get("results", [])] if not prior_urls else prior_urls
    slug, crawl, repo_res = repo_slug(repo_url), None, None
    if slug and not _issue_hit_of(repo_url, urls_seen):
        # Issues des Paper-Repos per Suche (bis 5 Treffer); der Crawl der neuesten Issues nur als Rueckfall, wenn auch das nichts lieferte
        repo_res = tools.search_repo_issues(slug, MAX_PAGES)
        calls.append("search")
        queries.append({"kind": "repo_search", "repo": slug})
        if not _issue_hit_of(repo_url, [r.get("url", "") for r in (repo_res or {}).get("results", [])]):
            crawl = tools.crawl_issues(repo_url, MAX_CRAWL)
            calls.append("crawl")
            queries.append({"kind": "crawl", "url": repo_url.rstrip("/") + "/issues"})
    exempt = (f"https://github.com/{slug.lower()}/",) if slug else ()   # Seiten des Paper-Repos brauchen den Titel nicht
    items = (res.get("results", []) + (method_res or {}).get("results", []) + (repo_res or {}).get("results", [])
             + list((crawl or {}).get("results", [])))
    docs = select_docs(items, title, arxiv_id, exempt, method, dataset, {r.get("url", "") for r in (method_res or {}).get("results", [])})
    kept, dropped = extract_objections(llm, docs[:MAX_CHUNKS], title, dataset, repo_files)
    n_before = len(kept)
    kept = [o for o in kept if o.source.lower().startswith(exempt) if exempt] + [
        o for o in kept if not (exempt and o.source.lower().startswith(exempt)) and names_paper(f"{o.statement} {o.quote}", title, arxiv_id, method)]
    dropped += n_before - len(kept)   # Aussage ohne Methode/Titel im Satz oder Zitat: zu allgemein
    findings = []
    for o in kept:
        kind = classify(o.statement, o.quote, o.protocol)
        findings.append(asdict(Finding(o.topic, o.statement, o.quote, o.source, kind, o.repo_check, KIND_LABEL[kind])))
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
    docs = select_docs(drop_noise(res, arxiv_id).get("results", []), title, arxiv_id)[:MAX_CHUNKS]
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

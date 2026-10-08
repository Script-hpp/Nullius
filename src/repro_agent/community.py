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
from .objections import PROTOCOL_KEYWORDS, Objection, extract_objections
from .setup_details import Doc, _chunks, _parse_list

CACHE_PATH = REPO_ROOT / "logs" / "community_cache.json"
CACHE_TTL_S = 7 * 86400
CACHE_VERSION = 3        # 3: generische Kritik-Vorlagen, Foren, Issue-Texte; 2: Seitenfilter, Methoden-Query, Fund muss Methode oder Titel nennen (aeltere Eintraege werden ignoriert)
MAX_PAGE_CHARS = 60_000  # laengere Seiten sind Sammlungen/Dumps, keine Diskussion
COLLECTION_RE = re.compile(r"gist\.github\.com|awesome[-_]|paper[-_]?club|reading[-_ ]?list|papers?[-_]we[-_]love|/awesome", re.I)
MAX_PAGES = 5            # Treffer der Suche
MAX_CRAWL = 3            # Seiten des Issue-Crawls (immer limit)
MAX_READ = 3             # Discovery-Treffer, die statt einer Suche gelesen werden
MAX_BODIES = 4           # Issue-Seiten, deren Text per Extract nachgeladen wird (ein Aufruf)
BODY_MIN_CHARS = 1500    # kuerzere Issue-Treffer gelten als Kopfzeilen ohne Text
ISSUE_PAGE_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull|discussions)/\d+/?$", re.I)
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


def _numbered_issue_hit(urls: list[str]) -> bool:
    """Eine einzelne Issue-/PR-/Diskussionsseite (mit Text), nicht die Listenseite `/issues`."""
    return any(ISSUE_PAGE_RE.match(u.split("#")[0].split("?")[0]) for u in urls)


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


SETUP_KEYWORDS = PROTOCOL_KEYWORDS + ("random seed", "seeds", "batch size", "learning rate", "input length", "lookback", "window size",
                                      "data split", "evaluation protocol")


def is_setup_topic(o: Objection) -> bool:
    """Evaluations-, Protokoll- oder Setup-Thema (messbare Klassen: Protokoll, Parameter, Seed): Stichwortpruefung auf Aussage, Zitat, Thema."""
    low = f"{o.statement} {o.quote} {o.topic}".lower()
    return o.protocol or any(k in low for k in SETUP_KEYWORDS)


def select_docs(items: list[dict], title: str, arxiv_id: str, exempt: tuple[str, ...] = (), method: str | None = None,
                dataset: str | None = None, method_hits: set[str] = frozenset()) -> list[Doc]:
    """Seiten, die gelesen werden: kein Duplikat, nicht leer, <= MAX_PAGE_CHARS, keine Sammelseite. Seiten des Paper-Repos (exempt) sind
    gesetzt; sonst muss der Text das Paper nennen (arXiv-ID oder voller Titel); Treffer der Methoden-Query (`method_hits`, URLs) auch mit
    Methodenname plus Datensatz im Text (Kritik-Papers nennen selten den Titel)."""
    docs, seen = [], set()
    kws = keywords(title, arxiv_id)
    for it in items:
        url, text = it.get("url", ""), it.get("raw_content") or it.get("content") or ""
        key = url.split("#")[0].split("?")[0].rstrip("/").lower()   # issues/14 und issues/14?timeline_page=1 sind dieselbe Seite
        if not text or key in seen or len(text) > MAX_PAGE_CHARS or is_collection(url):
            continue
        seen.add(key)
        low = text.lower()
        if exempt and url.lower().startswith(tuple(exempt)):
            docs.append(Doc(url, text))
        elif any(k in low for k in kws) or (url in method_hits and _word(text, method or "") and _word(text, dataset or "")):
            docs.append(Doc(url, text))
    return docs


def _pages(docs: list[Doc]) -> list[dict]:
    import hashlib
    return [{"url": d.name, "n_chars": len(d.text), "sha256": hashlib.sha256(d.text.encode()).hexdigest()} for d in docs]


def _with_issue_bodies(tools, items: list[dict], calls: list[str], queries: list[dict]) -> list[dict]:
    """Issue-Treffer mit kurzem Text (Kopfzeile statt Beitrag) bekommen ihren Text per Extract (ein Aufruf, hoechstens MAX_BODIES Seiten).
    Ersetzt den Text nur, wenn der Extract laenger ist; Fehler lassen die Treffer unveraendert."""
    short = []
    for it in items:
        u, text = it.get("url", ""), it.get("raw_content") or it.get("content") or ""
        if _numbered_issue_hit([u]) and len(text) < BODY_MIN_CHARS and u not in short:
            short.append(u)
    short = short[:MAX_BODIES]
    if not short:
        return items
    try:
        got = {r.get("url", ""): r.get("raw_content") or r.get("content") or "" for r in (tools.read_pages(short, MAX_BODIES) or {}).get("results", [])}
    except Exception as exc:   # bezahlte Treffer nicht wegen eines Extract-Fehlers verlieren
        if type(exc).__name__ == "BillingBlocked":
            raise
        return items
    calls.append("extract")
    queries.append({"kind": "issue_bodies", "urls": short})
    return [{**it, "raw_content": got[it["url"]]} if it.get("url") in got and len(got[it["url"]]) > len(it.get("raw_content") or it.get("content") or "") else it
            for it in items]


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
    crit_res = None
    if method and dataset and hasattr(tools, "search_critique"):   # generische Kritik-Vorlagen (Fehler, Neubewertung, Folgearbeiten, Foren)
        crit_res = drop_noise(tools.search_critique(method, dataset, MAX_PAGES), arxiv_id)
        calls.extend(["search"] * len(crit_res.get("queries", [])))
        queries.append({"kind": "critique_search", "method": method, "dataset": dataset, "queries": crit_res.get("queries", [])})
    urls_seen = [r.get("url", "") for r in res.get("results", [])] if not prior_urls else prior_urls
    slug, crawl, repo_res = repo_slug(repo_url), None, None
    if slug and not _issue_hit_of(repo_url, urls_seen):
        # Issues des Paper-Repos per Suche (bis 5 Treffer)
        repo_res = tools.search_repo_issues(slug, MAX_PAGES)
        calls.append("search")
        queries.append({"kind": "repo_search", "repo": slug})
    found = urls_seen + [r.get("url", "") for src in (method_res, crit_res, repo_res) for r in (src or {}).get("results", [])]
    if slug and not _numbered_issue_hit([u for u in found if u.lower().startswith(f"https://github.com/{slug.lower()}/")]):
        # Kein Issue mit Text gefunden (nur Listenseite oder gar nichts): Crawl der neuesten Issues, immer mit limit
        crawl = tools.crawl_issues(repo_url, MAX_CRAWL)
        calls.append("crawl")
        queries.append({"kind": "crawl", "url": repo_url.rstrip("/") + "/issues"})
    exempt = (f"https://github.com/{slug.lower()}/",) if slug else ()   # Seiten des Paper-Repos brauchen den Titel nicht
    items = (res.get("results", []) + (method_res or {}).get("results", []) + (crit_res or {}).get("results", [])
             + (repo_res or {}).get("results", []))
    if not prior_urls:   # Discovery-Treffer kamen schon per Extract
        items = _with_issue_bodies(tools, items, calls, queries)
    items += list((crawl or {}).get("results", []))
    method_hits = {r.get("url", "") for src in (method_res, crit_res) for r in (src or {}).get("results", [])}
    docs = select_docs(items, title, arxiv_id, exempt, method, dataset, method_hits)
    drops: list[dict] = []
    cands: list[dict] = []
    kept, dropped = extract_objections(llm, docs[:MAX_CHUNKS], title, dataset, repo_files, drops, cands)
    n_before = len(kept)
    all_kept = kept
    page_text = {d.name: d.text for d in docs}
    reasons: dict[int, str] = {}

    def passes(o: Objection) -> bool:
        if exempt and o.source.lower().startswith(exempt):
            return True
        if names_paper(f"{o.statement} {o.quote}", title, arxiv_id, method):
            return True
        if is_setup_topic(o) and names_paper(page_text.get(o.source, ""), title, arxiv_id, method):   # Seitenebene, nur Setup-Themen
            reasons[id(o)] = "page_names_method"
            return True
        return False

    kept = [o for o in kept if passes(o)]
    dropped += n_before - len(kept)   # Aussage ohne Methode/Titel im Satz oder Zitat: zu allgemein
    drops += [{"url": o.source, "reason": "names_neither_method_title_nor_arxiv_id", "topic": o.topic, "statement": o.statement[:300],
               "quote": o.quote[:300]} for o in all_kept if o not in kept]
    findings = []
    for o in kept:
        kind = classify(o.statement, o.quote, o.protocol)
        findings.append({**asdict(Finding(o.topic, o.statement, o.quote, o.source, kind, o.repo_check, KIND_LABEL[kind])),
                         "pass_reason": reasons.get(id(o), "statement_names_method")})
    for c in cands:   # Ausgang je roher Kandidat, damit nachvollziehbar bleibt, warum er blieb oder fiel
        k = next((o for o in all_kept if o.source == c["url"] and o.quote[:300] == c["quote"]), None)
        if not c["quote_in_page"]:
            c["outcome"] = "dropped_quote_not_in_page"
        elif k is None:
            c["outcome"] = "dropped_duplicate"
        else:
            c["outcome"] = "kept_" + reasons.get(id(k), "statement_names_method") if k in kept else "dropped_names_neither_method_title_nor_arxiv_id"
    hits = [{"url": r.get("url", ""), "title": r.get("title", ""), "n_chars": len(r.get("raw_content") or r.get("content") or "")}
            for r in items]   # was Tavily lieferte, vor dem Seitenfilter
    read = {d.name for d in docs}
    return {"findings": findings, "dropped": dropped, "pages": _pages(docs), "queries": queries,
            "reused_discovery": bool(prior_urls), "calls": calls, "hits": hits,
            "not_read": [h["url"] for h in hits if h["url"] not in read], "dropped_items": drops, "candidates": cands}


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

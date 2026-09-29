from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
import httpx

from .costs import BudgetExceeded
from .llm import BillingBlocked, LLMClient
from .tavily_tools import TavilyTools

MIN_FULLTEXT_CHARS = 800    # gezielter Extract liefert ca. 4000 Zeichen; darunter nur Reste (Live-Lauf 1: SVG-Namen)
MAX_LLM_TEXT_CHARS = 20000
SECTION_KEYWORDS = ("experiment", "setup", "setting", "result", "evaluation", "dataset", "benchmark", "implementation")


def _tasks_for(dataset: str) -> dict:
    """Alle Tasks eines Datensatzes, z. B. Cora -> Cora (Planetoid) und Cora-LCC (Shchur-Splits)."""
    from .tasks import TASKS
    return {k: v for k, v in TASKS.items() if k == dataset or k.startswith(dataset + "-")}


def _last_json_object(text: str) -> dict | None:
    """Letztes JSON-Objekt im Text; Lightning schreibt oft Denktext davor."""
    import re
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            value, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _select_sections(text: str, abstract: str, limit: int = MAX_LLM_TEXT_CHARS) -> str:
    """Abstract plus Experiment-/Setup-/Ergebnis-Abschnitte statt Anfang+Ende des Papers.

    Ohne erkennbare Markdown-Ueberschriften: Textanfang bis `limit`.
    """
    import re
    if len(text) <= limit:
        return text
    heads = list(re.finditer(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE))
    parts = []
    for i, h in enumerate(heads):
        if any(k in h.group(1).lower() for k in SECTION_KEYWORDS):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            parts.append(text[h.start():end])
    if not parts:
        return text[:limit]
    out = "Abstract: " + abstract
    for part in parts:
        if len(out) >= limit:
            break
        out += "\n\n" + part[: limit - len(out)]
    return out

@dataclass
class CandidatePaper:
    title: str
    arxiv_id: str
    dataset: str
    claim: dict
    reason: str
    github_url: str | None
    alternatives: list[dict] = __import__('dataclasses').field(default_factory=list)
    status: str = "ungeprueft"
    found_by: str = "arxiv"                 # "tavily" | "arxiv" (Ersatzweg)
    source_url: str | None = None           # Tavily-Treffer, ueber den das Paper gefunden wurde
    community: list[dict] = __import__('dataclasses').field(default_factory=list)   # Tavily-Hinweise, kein Urteil
    
    def to_dict(self):
        return {
            "status": self.status,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "dataset": self.dataset,
            "claim": self.claim,
            "reason": self.reason,
            "github_url": self.github_url,
            "alternatives": self.alternatives or [],
            "found_by": self.found_by,
            "source_url": self.source_url,
            "community": self.community or [],
        }


# Tavily-Suche pro Datensatz: aktuelle Papers mit eigenen Ergebnissen auf dem Benchmark.
DISCOVERY_QUERIES = {
    "Cora": "new graph neural network node classification Cora test accuracy results table",
    "ETTh1": "new long-term time series forecasting ETTh1 MSE MAE results table",
}
# Tavily-Extract-Frage: nur Abschnitte zu Protokoll und Ergebnis zurueckgeben.
EXTRACT_QUERIES = {
    "Cora": "Cora node classification: data split (Planetoid, 20 labels per class), evaluation protocol, test accuracy",
    "ETTh1": "ETTh1 forecasting: input length, prediction horizons 96/192/336/720, data split, test MSE and MAE",
}
ARXIV_ID_RE = __import__("re").compile(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})")


def _parse_atom(xml_text: str) -> list[dict]:
    import re
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', ns):
        id_raw = entry.find('atom:id', ns).text.split('/')[-1]
        papers.append({
            "arxiv_id": re.sub(r'v\d+$', '', id_raw),
            "title": " ".join(entry.find('atom:title', ns).text.split()),
            "abstract": " ".join(entry.find('atom:summary', ns).text.split()),
            "authors": [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)],
        })
    return papers


class DiscoveryAgent:
    def __init__(self, llm: LLMClient | None = None, tavily: TavilyTools | None = None, cache_path=None):
        self.cache_path = cache_path
        self.llm = llm or LLMClient()
        self.tavily = tavily or TavilyTools()

    def _tavily(self, method: str, role: int, endpoint: str, label: str, *args, top_title: str | None = None, **kwargs) -> dict:
        """Tavily-Aufruf mit Dauer und Treffern als Ereignis `tavily_call` fuer die Live-UI."""
        import time
        t0 = time.time()
        res = getattr(self.tavily, method)(*args, **kwargs)
        dur = int((time.time() - t0) * 1000)
        hits = res.get("results", []) if isinstance(res, dict) else []
        top_hits = [{"url": r.get("url"), "title": top_title or r.get("title")} for r in hits[:3]]
        self._emit_tavily(role, endpoint, label, len(hits), top_hits, dur)
        return res

    def _curate_repo(self, title):
        return self._tavily("curate_repo", 8, "search", title, title)

    def _emit_tavily(self, role, endpoint, query, hits, top_hits, dur):
        if not getattr(self, "_on_event", None):
            return
        self._on_event({"type": "tavily_call", "role": role, "endpoint": endpoint, "query": query, "hits": hits,
                        "top_hits": top_hits, "duration_ms": dur, "mode": "live"})

    def _find_papers_tavily(self, dataset: str) -> list[dict]:
        """Tavily-Suche -> arXiv-IDs; Metadaten (Titel, Abstract) danach kostenlos von der arXiv-API."""
        query = DISCOVERY_QUERIES.get(dataset)
        if not query or not hasattr(self.tavily, "curate_papers"):
            return []
        res = self._tavily("curate_papers", 8, "search", query, query)
        found = {}
        for r in res.get("results", []):
            m = ARXIV_ID_RE.search(r.get("url", ""))
            if m and m.group(1) not in found:
                found[m.group(1)] = {"source_url": r.get("url"), "snippet": r.get("content", "")}
        if not found:
            return []
        resp = httpx.get("https://export.arxiv.org/api/query",
                         params={"id_list": ",".join(found), "max_results": len(found)}, timeout=30.0)
        resp.raise_for_status()
        papers = _parse_atom(resp.text)
        for p in papers:
            p.update(found.get(p["arxiv_id"], {}), found_by="tavily")
        return papers

    def _search_arxiv(self, dataset: str, max_results: int = 10) -> list[dict]:
        """Ersatzweg ohne Tavily-Treffer: neueste arXiv-Papers, die den Datensatz nennen."""
        query = f'all:"{dataset}" AND (all:"node classification" OR all:"graph neural network" OR all:"time series" OR all:"forecasting" OR all:"recommendation")'
        resp = httpx.get("https://export.arxiv.org/api/query", params={
            "search_query": query, "start": 0, "max_results": max_results,
            "sortBy": "submittedDate", "sortOrder": "descending"}, timeout=30.0)
        resp.raise_for_status()
        return _parse_atom(resp.text)
        
    def _extract_claim(self, text: str, dataset: str, emit_func=None) -> dict | None:
        """LLM extrahiert einen Claim und prueft das Protokoll gegen alle Tasks dieses Datensatzes."""
        import time
        tasks = _tasks_for(dataset)
        protocols = "\n\n".join(f"[{name}]\n{factory().data_description}" for name, factory in tasks.items())
        
        system_prompt = (
            "You are an expert AI researcher. Extract ONE clear, numerical, testable claim "
            f"from the following text regarding the dataset '{dataset}'. "
            "You must also verify whether the paper follows one of the evaluation protocols we can replicate. "
            "Supported protocols, each with its task name in brackets:\n"
            f"{protocols}\n\n"
            "Reply with a JSON object containing keys: "
            "'quote' (a VERBATIM excerpt copied from the text that proves BOTH the claim and the protocol match), "
            "'metric' (e.g., accuracy, MSE), "
            "'target' (the numerical value as float), "
            "'protocol_match' (boolean: true if the paper's protocol matches one supported protocol), "
            "'matched_task' (the task name in brackets that matches, or null), "
            "and 'protocol_reason' (a short explanation citing the text to justify your protocol_match decision). "
            "If no clear claim or protocol is found, return {\"protocol_match\": false}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            t0 = time.time()
            # Lightning denkt im Antworttext; 4000 reichten im Live-Lauf nicht (finish_reason length).
            # Rolle extract (Super) wie Rollen 2/4: Lightning dachte im Live-Lauf 8000 Tokens ohne Antwort.
            resp = self.llm.complete("extract", messages, temperature=0.0, max_tokens=8000)
            is_truncated = getattr(resp, "finish_reason", "") == "length"
            if emit_func:
                emit_func({
                    "type": "llm_call",
                    "role": "extract",
                    "model": getattr(resp, "model", "unknown"),
                    "tokens": getattr(resp, "prompt_tokens", 0) + getattr(resp, "completion_tokens", 0),
                    "cost_usd": getattr(resp, "cost_usd", 0.0),
                    "duration_ms": int((time.time() - t0) * 1000)
                })
        except (BillingBlocked, BudgetExceeded):
            raise   # Abrechnung/Budget: Lauf abbrechen, nicht als Paper-Befund werten
        except Exception as e:
            print(f"LLM extract error: {e}")
            return {"llm_unreadable": True}
        # Abgeschnittene Antwort ist kein Befund ueber das Paper: vor dem Parsen pruefen.
        if is_truncated:
            return {"llm_truncated": True}
        claim_data = _last_json_object(resp.content or "")
        if claim_data is None:
            return {"llm_unreadable": True}
        return claim_data

    def _fetch_readme(self, repo_url: str) -> str:
        """Fetch README from github repo."""
        # Convert https://github.com/user/repo to raw URLs
        if not repo_url.startswith("https://github.com/"):
            return ""
        parts = repo_url.rstrip("/").split("/")
        if len(parts) < 5:
            return ""
        user, repo = parts[3], parts[4]
        
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/README.md"
            try:
                resp = httpx.get(raw_url, timeout=5.0)
                if resp.status_code == 200:
                    return resp.text
            except Exception:
                pass
        return ""

    def _find_repo_and_context(self, title: str, dataset: str, authors: list[str]) -> tuple[str | None, str, list[dict]]:
        """Find Github repos and score them based on author, paper link, and dataset."""
        import httpx
        
        candidates = []
        try:
            query = f'"{title}" github implementation'
            search_code = self._curate_repo(title)
            results = search_code.get("results", [])
            
            for r in results:
                url = r.get("url", "")
                if "github.com" in url:
                    readme = self._fetch_readme(url).lower()
                    if not readme:
                        continue
                        
                    keywords = [dataset.lower(), "cora", "citeseer", "pytorch geometric", "dgl", "ogb", "ett", "movielens"]
                    import re
                    if not any(re.search(r"\b" + re.escape(k) + r"\b", readme) for k in keywords):
                        continue
                        
                    # Valid candidate found
                    
                    # 1. Author-Match
                    author_match = False
                    owner_and_readme = url.split("github.com/")[-1].split("/")[0].lower() + " " + readme
                    for a in authors:
                        last_name = a.split(" ")[-1].lower()
                        if last_name and len(last_name) > 2 and last_name in owner_and_readme:
                            author_match = True
                            break
                            
                    # 2. Paper-Link
                    paper_link = False
                    if title.lower() in readme:
                        paper_link = True
                        
                    # 3. Dataset-Match (exact)
                    dataset_match = False
                    if dataset.lower() in readme:
                        dataset_match = True
                        
                    score = int(author_match) + int(paper_link) + int(dataset_match)
                    
                    # Get stars
                    stars = 0
                    try:
                        repo_path = url.split("github.com/")[-1]
                        if repo_path.endswith("/"):
                            repo_path = repo_path[:-1]
                        parts = repo_path.split("/")
                        if len(parts) >= 2:
                            api_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}"
                            resp = httpx.get(api_url, timeout=3.0)
                            if resp.status_code == 200:
                                stars = resp.json().get("stargazers_count", 0)
                    except Exception:
                        pass
                        
                    candidates.append({
                        "url": url,
                        "score": score,
                        "stars": stars,
                        "author_match": author_match,
                        "paper_link": paper_link,
                        "dataset_match": dataset_match
                    })
        except Exception as e:
            print(f"Tavily search error: {e}")
            
        if not candidates:
            return None, "No valid GitHub repository found.", []
            
        seen_repos = set()
        deduped = []
        for c in candidates:
            repo_path = c["url"].split("github.com/")[-1].rstrip("/").lower()
            if repo_path not in seen_repos:
                seen_repos.add(repo_path)
                deduped.append(c)
        candidates = deduped
        
        # Sort by score desc, then stars desc
        candidates.sort(key=lambda x: (x["score"], x["stars"]), reverse=True)
        
        chosen = candidates[0]
        chosen_url = chosen["url"]
        
        # Check for controversy for the chosen one
        reason = f"Chosen over {len(candidates)-1} alternatives (Score: {chosen['score']}/3)." if len(candidates) > 1 else "Code found, validated via README."
            
        alternatives = [{"url": c["url"], "summary": c.get("content", "")[:200]} for c in candidates[1:4]]
        return chosen_url, reason, alternatives
        
    def discover(self, datasets: list[str], max_papers: int = 10, on_event=None) -> list[CandidatePaper]:
        """Run the full discovery pipeline for the given datasets."""
        import time
        seq = 0
        def emit(evt: dict):
            nonlocal seq
            evt["ts"] = time.time()
            evt["seq"] = seq
            seq += 1
            if on_event:
                on_event(evt)
        self._on_event = emit
        # Tavily-Aufrufe meldet der Wrapper _tavily mit Dauer und Treffern.
        # on_call behaelt die 4-Argument-Signatur von loop.py.
        self.tavily.on_call = None

        candidates = []

        # Load cache to save Tavily credits
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_file = self.cache_path or os.path.join(repo_root, "logs", "discovery_cache.json")
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        def _update_cache(key, data):
            import os, json
            c = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f: c = json.load(f)
                except Exception: pass
            c[key] = data
            os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
            tmp = cache_file + ".tmp"
            with open(tmp, "w") as f: json.dump(c, f, indent=2)
            os.replace(tmp, cache_file)
            cache[key] = data


        if os.environ.get("TAVILY_MODE") == "fixture":
            emit({"type": "error", "message": "Discovery braucht live"})
            return []

        for ds in datasets:
            emit({"type": "pending", "dataset": ds, "message": "Searching papers with Tavily..."})
            try:
                papers = self._find_papers_tavily(ds)
            except (BillingBlocked, BudgetExceeded):
                raise
            except Exception as e:
                emit({"type": "error", "dataset": ds, "message": f"Tavily paper search error: {e}"})
                papers = []
            if not papers:
                # Ersatzweg: arXiv-API (kostenlos), wenn Tavily nichts Brauchbares findet.
                emit({"type": "pending", "dataset": ds, "message": "No Tavily hits, falling back to arXiv..."})
                t0 = time.time()
                try:
                    papers = self._search_arxiv(ds, max_results=30)
                except Exception as e:
                    emit({"type": "error", "dataset": ds, "message": f"arXiv API error: {e}"})
                    continue
                emit({"type": "arxiv_call", "dataset": ds, "duration_ms": int((time.time() - t0) * 1000)})
            if not papers:
                emit({"type": "reject", "dataset": ds, "reason": "No results found on arXiv", "stage": "prefilter"})
                continue
                
            dataset_candidates = 0
            extracts_count = 0
            for p in papers:
                if dataset_candidates >= 2:
                    break
                cache_key = p["arxiv_id"]
                if cache_key in cache:
                    cached_data = cache[cache_key]
                    if cached_data.get("status") == "rejected": continue
                    import dataclasses
                    valid_keys = {f.name for f in dataclasses.fields(CandidatePaper)}
                    filtered_data = {k: v for k, v in cached_data.items() if k in valid_keys}
                    candidates.append(CandidatePaper(**filtered_data))
                    dataset_candidates += 1
                    continue
                    
                # Tavily-Ausschnitt zaehlt mit: Ergebnisse stehen oft nur in Tabellen, nicht im Abstract.
                abstract = (p["abstract"] + " " + p.get("snippet", "")).lower()
                import re
                if ds.lower() not in abstract or not re.search(r"\d+\.\d+|\d+%", abstract):
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "Regex pre-filter failed", "stage": "prefilter"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": "Regex pre-filter failed", "stage": "prefilter"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                html_url = f"https://arxiv.org/html/{cache_key}"
                
                if extracts_count >= 3:
                    emit({"type": "warning", "dataset": ds, "message": "Max 3 Tavily extracts reached."})
                    break
                    
                emit({"type": "extracting", "arxiv_id": cache_key, "title": p["title"], "dataset": ds})
                extracts_count += 1
                try:
                    text = ""
                    for _attempt in range(2):
                        html_data = self._tavily("curate_extract", 2, "extract", html_url, html_url,
                                                 EXTRACT_QUERIES.get(ds, ds), top_title=p["title"])
                        results = html_data.get("results") or [{}]
                        text = results[0].get("raw_content") or ""
                        if len(text) >= MIN_FULLTEXT_CHARS:
                            break
                except Exception:
                    text = ""

                # Ohne Volltext ist das Protokoll nicht beurteilbar (Abstract reicht nicht).
                # Nicht cachen: ein spaeterer Lauf kann den Volltext bekommen.
                if len(text) < MIN_FULLTEXT_CHARS:
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "Volltext nicht lesbar", "stage": "extract", "text_length": len(text)})
                    continue

                text = _select_sections("Abstract: " + p["abstract"] + "\n\n" + text, p["abstract"])

                claim = self._extract_claim(text, ds, emit_func=emit)
                claim_data = claim or {}

                # LLM-Probleme sind kein Befund ueber das Paper: nicht cachen.
                if claim_data.get("llm_truncated"):
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "llm_truncated", "stage": "extract"})
                    continue
                if claim_data.get("llm_unreadable"):
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "LLM-Antwort nicht lesbar", "stage": "extract"})
                    continue

                if not claim_data:
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "No numerical claim found", "stage": "extract"})
                    _update_cache(cache_key, {"status": "rejected", "reason": "No numerical claim found"})
                    continue
                    
                # Prompt zeigt Tasknamen in Klammern; Live-Lauf 4: LLM antwortete "[Cora-LCC]"
                if isinstance(claim_data.get("matched_task"), str):
                    claim_data["matched_task"] = claim_data["matched_task"].strip().strip("[]").strip()
                # matched_task muss ein echter Task sein, sonst gilt der Match nicht
                # (Formfehler des LLM, kein Befund ueber das Paper: nicht cachen)
                if claim_data.get("protocol_match") and claim_data.get("matched_task") not in _tasks_for(ds):
                    emit({"type": "reject", "arxiv_id": cache_key, "stage": "protocol",
                          "reason": f"LLM nannte keinen unterstuetzten Task ({claim_data.get('matched_task')!r})"})
                    continue
                if not claim_data.get("protocol_match"):
                    reason_msg = claim_data.get("protocol_reason", "Protocol mismatch")
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason_msg, "stage": "protocol"})
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_msg})
                    continue
                    
                quote = claim_data.get("quote", "")
                if not quote:
                    reason = "No quote provided by LLM"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                import re
                def norm_ws(s): return re.sub(r"\s+", " ", s.replace("\n", " ")).strip().lower()
                
                if norm_ws(quote) not in norm_ws(text):
                    reason = "Quote not found in text"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                try:
                    target_raw = claim_data.get("target")
                    if target_raw is None:
                        raise ValueError("No target provided")
                    if isinstance(target_raw, str):
                        target_raw = target_raw.replace('%', '')
                    target_val = float(target_raw)
                except (ValueError, TypeError):
                    reason = "Target missing or not parsable"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                printed = re.findall(r"\d+(?:\.\d+)?", norm_ws(quote))
                if not any(
                    abs(float(p) - target_val) < 1e-9 or 
                    abs(float(p) / 100 - target_val) < 1e-9 or 
                    abs(float(p) * 100 - target_val) < 1e-9 
                    for p in printed
                ):
                    reason = "Target value not found numerically in quote"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                claim_data["protocol_evidence"] = "Protokoll: LLM-Einschaetzung"
                claim = claim_data
                    
                emit({"type": "searching", "arxiv_id": cache_key, "claim": {"text": json.dumps(claim), "target": target_val}})
                    
                github_url, reason, alternatives = self._find_repo_and_context(p["title"], ds, p.get("authors", []))
                if not github_url:
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "No valid GitHub repo matching dataset found.", "stage": "repo"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": "No valid GitHub repo matching dataset found.", "stage": "repo"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                
                # Community-Hinweise (Issues, Reviews): nur als Belege, das Urteil faellt im Loop.
                community = []
                if hasattr(self.tavily, "curate_discussion"):
                    try:
                        disc = self._tavily("curate_discussion", 8, "search", p["title"], p["title"])
                        community = [{"url": r.get("url"), "title": r.get("title"), "snippet": (r.get("content") or "")[:300]}
                                     for r in disc.get("results", [])]
                    except (BillingBlocked, BudgetExceeded):
                        raise
                    except Exception as e:
                        emit({"type": "warning", "arxiv_id": cache_key, "message": f"Community search failed: {e}"})

                emit({"type": "candidate", "arxiv_id": cache_key, "reason": reason, "found_by": p.get("found_by", "arxiv"),
                      "community_hits": len(community)})
                
                paper = CandidatePaper(
                    title=p["title"],
                    arxiv_id=p["arxiv_id"],
                    dataset=ds,
                    claim=claim,
                    reason=reason,
                    github_url=github_url,
                    alternatives=alternatives,
                    found_by=p.get("found_by", "arxiv"),
                    source_url=p.get("source_url"),
                    community=community,
                )
                candidates.append(paper)
                _update_cache(cache_key, paper.to_dict())
                dataset_candidates += 1

        return candidates

if __name__ == "__main__":
    agent = DiscoveryAgent()
    print("Starting discovery for Cora, ETTh1...")
    results = agent.discover(["Cora", "ETTh1"])
    for r in results:
        print(json.dumps(r.to_dict(), indent=2))

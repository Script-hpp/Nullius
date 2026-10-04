from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
import httpx

from .costs import BudgetExceeded
from .llm import BillingBlocked, LLMClient
from .claims_source import _norm as _table_norm
from .table_check import check_columns
from .tavily_tools import TavilyTools

MIN_FULLTEXT_CHARS = 800    # gezielter Extract liefert ca. 4000 Zeichen; darunter nur Reste (Live-Lauf 1: SVG-Namen)
MAX_LLM_TEXT_CHARS = 20000
SECTION_KEYWORDS = ("experiment", "setup", "setting", "result", "evaluation", "dataset", "benchmark", "implementation")


def _tasks_for(dataset: str) -> dict:
    """Alle Tasks eines Datensatzes, z. B. Cora -> Cora (Planetoid) und Cora-LCC (Shchur-Splits)."""
    from .tasks import TASKS
    return {k: v for k, v in TASKS.items() if k == dataset or k.startswith(dataset + "-")}


def _normalize_repo_url(url: str) -> str | None:
    """https://github.com/owner/repo/blob/... -> https://github.com/owner/repo, sonst None."""
    import re
    m = re.match(r"https?://(?:www\.)?github\.com/([^/?#\s]+)/([^/?#\s]+)", url or "")
    if not m:
        return None
    repo = m.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"https://github.com/{m.group(1)}/{repo}"


ETT_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")   # Horizont 96, MSE/MAE; tasks.py ETT
ALLOWED_METRICS = {"Cora": {"accuracy"}, "CiteSeer": {"accuracy"}, "PubMed": {"accuracy"},
                   **{ds: {"mse", "mae"} for ds in ETT_DATASETS}}
METRIC_ALIASES = {"accuracy": "accuracy", "acc": "accuracy", "test accuracy": "accuracy",
                  "mse": "mse", "mean squared error": "mse", "mae": "mae", "mean absolute error": "mae"}


def _valid_input_length(value, text: str) -> int | None:
    """Eingabelaenge (1..720) nur, wenn die Zahl im Text vorkommt; sonst None (nie raten)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= n <= 720 or isinstance(value, bool):
        return None
    return n if re.search(rf"(?<![\d.]){n}(?![\d.])", text) else None


def _normalize_metric(metric) -> str | None:
    return METRIC_ALIASES.get(str(metric or "").strip().lower())


def _norm_quote(s: str) -> str:
    """Zitatvergleich: Kleinschreibung, Tabellen-Pipes und $ weg, \\pm / +- / +/- einheitlich, Leerraum zusammengefasst."""
    import re
    s = s.lower().replace("\\pm", "±").replace("+/-", "±").replace("+-", "±")
    s = s.replace("|", " ").replace("$", "").replace("±", " ± ")
    return re.sub(r"\s+", " ", s).strip()


METHOD_CONTEXT_CHARS = 2000   # Tabellen nennen die Methode im Spaltenkopf, nicht in der Zeile
TITLE_STOPWORDS = {"with", "from", "that", "this", "towards", "toward", "using", "based", "into", "over", "their", "than", "have", "more"}
COMMUNITY_TITLE_SHARE = 0.6


def _quote_lines(quote: str) -> list[str]:
    """Nicht-leere Zitatzeilen (normalisiert); Trennzeilen wie |---| fallen weg."""
    return [n for n in (_norm_quote(l) for l in quote.split("\n")) if n]


def _quote_in_text(quote: str, text: str) -> bool:
    """Zeilenweise: jede nicht-leere Zeile des Zitats muss fuer sich im Text stehen (Kopfzeile darf weiter vorn stehen).
    Einzeilige Zitate wie bisher; kein Fuzzy-Match."""
    t = _norm_quote(text)
    lines = _quote_lines(quote)
    return bool(lines) and all(l in t for l in lines)


def _number_line(quote: str, target) -> str | None:
    """Rohzeile des Zitats, die die Zielzahl enthaelt (Zahl, Prozent oder Anteil)."""
    import re
    for line in quote.split("\n"):
        for m in re.finditer(r"\d+(?:\.\d+)?", _norm_quote(line)):
            n = float(m.group())
            if any(abs(x - target) < 1e-9 for x in (n, n / 100, n * 100)):
                return line
    return None


def _method_supported(method, quote: str, text: str, anchor: str | None = None) -> bool:
    """Methode steht im Zitat oder in den 2000 Zeichen vor der Fundstelle der Zeile mit der Zahl (Spaltenkopf)."""
    if not (isinstance(method, str) and method.strip()):
        return False
    m, q, t = _norm_quote(method), _norm_quote(quote), _norm_quote(text)
    if m in q:
        return True
    pos = t.find(_norm_quote(anchor) if anchor else q)
    return pos >= 0 and m in t[max(0, pos - METHOD_CONTEXT_CHARS):pos]


def _community_related(title: str, arxiv_id: str, hit_title: str, snippet: str) -> bool:
    """Treffer gehoert zum Paper: Titel oder arXiv-ID im Treffer, oder mind. 60 % der Titelwoerter."""
    import re
    def alnum(x): return re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()
    hay = alnum(f"{hit_title} {snippet}")
    if arxiv_id and arxiv_id.lower() in (f"{hit_title} {snippet}").lower():
        return True
    if alnum(title) and alnum(title) in hay:
        return True
    words = {w for w in alnum(title).split() if len(w) > 3 and w not in TITLE_STOPWORDS}
    if not words:
        return False
    return len(words & set(hay.split())) / len(words) >= COMMUNITY_TITLE_SHARE


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
    protocol: str = "unklar"                # "passt" (Zitat belegt) | "unklar" (Protokoll nicht belegt)

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
            "protocol": self.protocol,
        }


# Tavily-Suche pro Datensatz: hoechstens 2 Queries, je eine pro Task, mit den Begriffen unseres Protokolls.
DISCOVERY_QUERIES = {
    "Cora": [
        "node classification Cora Planetoid public split 140 training nodes test accuracy results table",
        "node classification Cora random splits Shchur largest connected component test accuracy results table",
    ],
    "ETTh1": [
        "long-term time series forecasting ETTh1 prediction length 96 192 336 720 12/4/4 months split MSE MAE results table",
    ],
    "CiteSeer": [
        "node classification CiteSeer Planetoid public split 120 training nodes test accuracy results table",
    ],
    "PubMed": [
        "node classification PubMed Planetoid public split 60 training nodes test accuracy results table",
    ],
    "ETTh2": [
        "long-term time series forecasting ETTh2 prediction length 96 192 336 720 12/4/4 months split MSE MAE results table",
    ],
    "ETTm1": [
        "long-term time series forecasting ETTm1 prediction length 96 192 336 720 12/4/4 months split MSE MAE results table",
    ],
    "ETTm2": [
        "long-term time series forecasting ETTm2 prediction length 96 192 336 720 12/4/4 months split MSE MAE results table",
    ],
}
MAX_SEARCHES_PER_DATASET = 2
# Tavily-Extract-Frage: nur Abschnitte zu Protokoll und Ergebnis zurueckgeben.
EXTRACT_QUERIES = {
    "Cora": "Cora node classification: data split (Planetoid public split 140 training nodes, or random splits / largest connected component), evaluation protocol, test accuracy",
    "ETTh1": "ETTh1 forecasting: input length (look-back window), prediction horizons 96/192/336/720, data split 12/4/4 months, test MSE and MAE",
    "CiteSeer": "CiteSeer node classification: data split (Planetoid public split 120 training nodes, 500 validation, 1000 test), evaluation protocol, test accuracy",
    "PubMed": "PubMed node classification: data split (Planetoid public split 60 training nodes, 500 validation, 1000 test), evaluation protocol, test accuracy",
    **{ds: f"{ds} forecasting: input length (look-back window), prediction horizons 96/192/336/720, data split 12/4/4 months, test MSE and MAE"
       for ds in ("ETTh2", "ETTm1", "ETTm2")},
}
CACHE_VERSION = 3   # 3: Feld claim.input_length, Hauptergebnis statt Variante; 2: dreistufiges Protokoll (Katalog-Import liest 2 und 3)
# Aequivalente Angaben, damit ein Paper mit anderer Schreibweise desselben Protokolls nicht als abweichend gilt.
DISCOVERY_PROTOCOL_HINTS = {
    "ETTh1": ("Equivalent statements: 12/4/4 months; 8640/2880/2880 rows; papers often report window/sample counts instead "
              "(e.g. 8545/2881/2881 as quoted in arXiv 2602.00717, or other counts slightly below 8640/2880/2880 "
              "depending on input length and horizon) - these are the same split. "
              "Any input length up to 720 is supported; only the horizon 96 is fixed."),
    "Cora": "Planetoid public split = 140/500/1000 = 20 labels per class.",
    "CiteSeer": "Planetoid public split = 120/500/1000 = 20 labels per class.",
    "PubMed": "Planetoid public split = 60/500/1000 = 20 labels per class.",
    **{ds: ("Equivalent statements: 12/4/4 months; for 15-minute data (ETTm) 34560/11520/11520 rows, for hourly data (ETTh) 8640/2880/2880 rows; "
            "papers often report window/sample counts instead - these are the same split. "
            "Any input length up to 720 is supported; only the horizon 96 is fixed.") for ds in ("ETTh2", "ETTm1", "ETTm2")},
    "Cora-LCC": "Shchur et al. 2018 random splits, 20 per class train, 30 per class val, largest connected component.",
}
MAX_README_ARXIV_LINKS = 5
MAX_README_GITHUB_LINKS = 10
# Kennwoerter unserer Protokolle: nennt ein passt_nicht-Grund eines davon, hat das LLM vermutlich nur gegen einen
# Task verglichen (Live-Lauf: Shchur-Splits gegen [Cora] statt [Cora-LCC]) -> nicht ablehnen.
TASK_KEYWORDS = {
    "Cora": ["planetoid", "public split"],
    "Cora-LCC": ["shchur", "largest connected component", "random split"],
    "CiteSeer": ["planetoid", "public split"],
    "PubMed": ["planetoid", "public split"],
    **{ds: ["12/4/4", "96"] for ds in ETT_DATASETS},
}
RETRY_TEMPERATURE = 0.3
ETTH1_HORIZON = 96   # tasks.py: der Task misst nur Horizont 96
NO_REPO_REASON = "kein passendes Repo gefunden"
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

    def _tavily(self, method: str, role: int, endpoint: str, label: str, *args, top_title: str | None = None,
                purpose: str | None = None, **kwargs) -> dict:
        """Tavily-Aufruf mit Dauer und Treffern als Ereignis `tavily_call` fuer die Live-UI.

        purpose: find_papers | read_paper | find_code | community (Ueberschrift in der UI).
        """
        import time
        t0 = time.time()
        res = getattr(self.tavily, method)(*args, **kwargs)
        dur = int((time.time() - t0) * 1000)
        hits = res.get("results", []) if isinstance(res, dict) else []
        top_hits = [{"url": r.get("url"), "title": top_title or r.get("title")} for r in hits[:3]]
        arxiv_ids = None
        if purpose == "find_papers":
            arxiv_ids = list(dict.fromkeys(m.group(1) for r in hits if (m := ARXIV_ID_RE.search(r.get("url", "")))))
        self._emit_tavily(role, endpoint, label, len(hits), top_hits, dur, purpose, arxiv_ids)
        return res

    def _curate_repo(self, title):
        return self._tavily("curate_repo", 8, "search", title, title, purpose="find_code")

    def _emit_tavily(self, role, endpoint, query, hits, top_hits, dur, purpose=None, arxiv_ids=None):
        if not getattr(self, "_on_event", None):
            return
        evt = {"type": "tavily_call", "role": role, "endpoint": endpoint, "query": query, "hits": hits,
               "top_hits": top_hits, "duration_ms": dur, "mode": "live"}
        if purpose:
            evt["purpose"] = purpose
        if arxiv_ids is not None:
            evt["arxiv_ids"] = arxiv_ids
        self._on_event(evt)

    def _find_papers_tavily(self, dataset: str) -> list[dict]:
        """Tavily-Suche -> arXiv-IDs; Metadaten (Titel, Abstract) danach kostenlos von der arXiv-API."""
        queries = DISCOVERY_QUERIES.get(dataset, [])[:MAX_SEARCHES_PER_DATASET]
        if not queries or not hasattr(self.tavily, "curate_papers"):
            return []
        per_query = []
        for query in queries:
            res = self._tavily("curate_papers", 8, "search", query, query, purpose="find_papers")
            per_query.append(res.get("results", []))
        # Treffer der Queries abwechselnd einreihen und deduplizieren: der Extract-Deckel trifft so beide Protokolle.
        found = {}
        for i in range(max((len(r) for r in per_query), default=0)):
            for results in per_query:
                if i < len(results):
                    r = results[i]
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
        
    def _extract_claim(self, text: str, dataset: str, emit_func=None, temperature: float = 0.0) -> dict | None:
        """LLM extrahiert einen Claim und prueft das Protokoll gegen alle Tasks dieses Datensatzes."""
        import time
        tasks = _tasks_for(dataset)
        protocols = "\n\n".join(
            f"[{name}]\n{factory().data_description}"
            + (f"\n{DISCOVERY_PROTOCOL_HINTS[name]}" if name in DISCOVERY_PROTOCOL_HINTS else "")
            for name, factory in tasks.items())
        
        system_prompt = (
            "You are an expert AI researcher. Extract ONE clear, numerical, testable claim "
            f"from the following text regarding the dataset '{dataset}'. "
            "You must also judge whether the paper evaluates with one of the protocols we can replicate. "
            "Supported protocols, each with its task name in brackets:\n"
            f"{protocols}\n\n"
            "The claim must be the ABSOLUTE test metric of the method PROPOSED in the paper, "
            "never a baseline and never a relative improvement (no 'reduction percent'). "
            "Choose the best or the highlighted result of the main configuration of the proposed method; "
            "never an ablation and never a restricted variant (for example 'frozen', 'w/o ...', 'without ...', 'linear probe'). "
            "Reply with keys: "
            "'method' (name of the proposed method exactly as written in the quote), "
            "'quote' (a VERBATIM excerpt copied from the text that names the method and gives the number), "
            "'metric' (accuracy for node classification; mse or mae for forecasting), "
            "'target' (the numerical value as float), "
            + ("'horizon' (integer prediction horizon of the number; choose the number for horizon 96, "
               "never an average over horizons, and the quote must contain 96), "
               "'input_length' (integer input length / look-back window used for that number, at most 720, "
               "or null if the text does not state it), " if dataset in ETT_DATASETS else "") +
            "'protocol' (one of \"passt\", \"unklar\", \"passt_nicht\"), "
            "'matched_task' (the task name in brackets when protocol is \"passt\", else null), "
            "'protocol_quote' (a VERBATIM excerpt copied from the text that shows the protocol; required for \"passt\" and \"passt_nicht\"), "
            "and 'protocol_reason' (one short sentence). "
            "Rules for 'protocol': "
            "\"passt\" = the text names the split, input length or horizons of one supported protocol; a standard split counts as matching when the text names it. "
            "\"passt_nicht\" = the text states a protocol that matches NONE of the supported protocols above (for example 60/20/20 splits or other input lengths); "
            "if it matches another supported protocol, answer \"passt\" with that task. Quote the stated protocol in protocol_quote. "
            "\"unklar\" = the text does not mention the protocol or does not state it clearly. "
            "Not mentioned means \"unklar\", never \"passt_nicht\". "
            "If you find no clear numerical claim, return an empty JSON object {}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            t0 = time.time()
            # Lightning denkt im Antworttext; 4000 reichten im Live-Lauf nicht (finish_reason length).
            # Rolle extract (Super) wie Rollen 2/4: Lightning dachte im Live-Lauf 8000 Tokens ohne Antwort.
            resp = self.llm.complete("extract", messages, temperature=temperature, max_tokens=8000)
            is_truncated = getattr(resp, "finish_reason", "") == "length"
            cached = bool(getattr(resp, "cached", False))
            if emit_func:
                emit_func({
                    "type": "llm_call",
                    "role": "extract",
                    "model": getattr(resp, "model", "unknown"),
                    "tokens": getattr(resp, "prompt_tokens", 0) + getattr(resp, "completion_tokens", 0),
                    "cost_usd": 0.0 if cached else getattr(resp, "cost_usd", 0.0),   # Cache-Treffer kosten nichts
                    "cached": cached,
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

    def _judge_protocol(self, claim_data: dict, text: str, dataset: str, norm_ws) -> tuple[str, str]:
        """Dreistufiges Protokollurteil, vom Code gegen den Text geprueft.

        `passt` braucht ein Zitat im Text und einen echten Task, `passt_nicht` (Reject) ein Zitat im Text.
        Fehlt der Beleg, gilt `unklar`: kein Befund ueber das Paper, kein Reject.
        """
        level = str(claim_data.get("protocol") or "").strip().lower()
        pq = claim_data.get("protocol_quote")
        quote_ok = isinstance(pq, str) and bool(pq.strip()) and norm_ws(pq) in norm_ws(text)
        if level == "passt":
            if not quote_ok:
                return "unklar", "Protokoll-Zitat fehlt oder steht nicht im Text"
            if claim_data.get("matched_task") not in _tasks_for(dataset):
                return "unklar", f"kein unterstuetzter Task ({claim_data.get('matched_task')!r})"
            return "passt", ""
        if level == "passt_nicht":
            if not quote_ok:
                return "unklar", "Abweichendes Protokoll ohne gueltiges Zitat"
            import re
            said = f"{pq} {claim_data.get('protocol_reason') or ''}".lower()
            for task in _tasks_for(dataset):
                for kw in TASK_KEYWORDS.get(task, []):
                    if re.search(r"(?<![\w/])" + re.escape(kw) + (r"(?![\w/])" if kw[-1].isdigit() else ""), said):
                        return "unklar", f"Grund nennt ein unterstuetztes Protokoll ({task}: {kw!r})"
            return "passt_nicht", ""
        return "unklar", ""

    @staticmethod
    def _candidate_event(paper: "CandidatePaper", cached: bool) -> dict:
        """Ereignis `candidate` mit allem, was die UI fuer die Karte braucht (Titel, Claim als Objekt, Code-Link)."""
        return {"type": "candidate", "arxiv_id": paper.arxiv_id, "cached": cached, "title": paper.title,
                "dataset": paper.dataset, "claim": paper.claim, "reason": paper.reason, "found_by": paper.found_by,
                "community_hits": len(paper.community), "protocol": paper.protocol,
                "github_url": paper.github_url, "source_url": paper.source_url}

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

    def _find_repo_and_context(self, title: str, dataset: str, authors: list[str], arxiv_id: str = "") -> tuple[str | None, str, list[dict]]:
        """Find Github repos and score them based on author, paper link, and dataset."""
        import httpx
        
        candidates = []
        try:
            query = f'"{title}" github implementation'
            search_code = self._curate_repo(title)
            results = search_code.get("results", [])
            
            import re
            def norm_title(t): return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
            for r in results:
                url = _normalize_repo_url(r.get("url", ""))
                if url:
                    readme = self._fetch_readme(url).lower()
                    if not readme:
                        continue

                    # Sammlung (Awesome-/Paper-Liste) ist keine Implementierung, auch wenn sie den Titel nennt
                    if (len(re.findall(r"arxiv\.org/(?:abs|pdf|html)/", readme)) > MAX_README_ARXIV_LINKS
                            or len(re.findall(r"github\.com/", readme)) > MAX_README_GITHUB_LINKS):
                        continue

                    keywords = [dataset.lower(), "cora", "citeseer", "pytorch geometric", "dgl", "pubmed", "ogb", "ett", "movielens"]
                    if not any(re.search(r"\b" + re.escape(k) + r"\b", readme) for k in keywords):
                        continue

                    # 1. Autor-Match: Owner-Name oder Autorenzeile, nicht beliebiges Vorkommen im README
                    owner = url.split("github.com/")[-1].split("/")[0].lower()
                    author_lines = " ".join(l for l in readme.splitlines() if "author" in l or l.strip().startswith("by "))
                    author_match = False
                    for a in authors:
                        last_name = a.split(" ")[-1].lower()
                        if last_name and len(last_name) > 2 and (last_name in owner or last_name in author_lines):
                            author_match = True
                            break

                    # 2. Paper-Link: arXiv-ID (ohne Version) oder normalisierter Titel im README
                    paper_link = bool(arxiv_id and arxiv_id.lower() in readme) or (
                        bool(norm_title(title)) and norm_title(title) in norm_title(readme))

                    # 3. Datensatz im README
                    dataset_match = dataset.lower() in readme

                    score = int(author_match) + int(paper_link) + int(dataset_match)
                    # Datensatz allein ist kein Beleg (Live-Lauf: TTS-Repo, Sammel-Repo)
                    if not (paper_link or (author_match and dataset_match)):
                        continue

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
            return None, NO_REPO_REASON, []
            
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
                    loaded = json.load(f)
                cache = {k: v for k, v in loaded.items()
                         if isinstance(v, dict) and v.get("cache_version") == CACHE_VERSION}
            except Exception:
                pass
                
        def _update_cache(key, data):
            import os, json
            c = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f: c = json.load(f)
                except Exception: pass
            data = {**data, "cache_version": CACHE_VERSION}
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
                    cached_paper = CandidatePaper(**filtered_data)
                    candidates.append(cached_paper)
                    dataset_candidates += 1
                    emit(self._candidate_event(cached_paper, cached=True))
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
                                                 EXTRACT_QUERIES.get(ds, ds), top_title=p["title"], purpose="read_paper")
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
                # Leeres Ergebnis ({} oder unlesbar) schwankt zwischen Laeufen: genau ein zweiter Versuch mit anderer
                # Temperatur (gleiche Temperatur kaeme aus dem Cache). Hoechstens eine Wiederholung pro Paper.
                if not claim_data or claim_data.get("llm_unreadable"):
                    claim_data = self._extract_claim(text, ds, emit_func=emit, temperature=RETRY_TEMPERATURE) or {}

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
                import re
                norm_ws = _norm_quote

                protocol, protocol_note = self._judge_protocol(claim_data, text, ds, norm_ws)
                if protocol == "passt_nicht":
                    reason_msg = claim_data.get("protocol_reason") or "Protocol mismatch"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason_msg, "stage": "protocol"})
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_msg})
                    continue
                claim_data["protocol"] = protocol
                if protocol == "unklar":
                    claim_data["matched_task"] = None
                    if protocol_note:
                        claim_data["protocol_reason"] = protocol_note

                quote = claim_data.get("quote", "")
                if not quote:
                    reason = "No quote provided by LLM"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"})
                    reason_str = {"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "quote"}.get("reason", "")
                    _update_cache(cache_key, {"status": "rejected", "reason": reason_str})
                    continue
                    
                if not _quote_in_text(quote, text):
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
                    
                if ds in ETT_DATASETS:
                    try:
                        horizon_ok = int(claim_data.get("horizon")) == ETTH1_HORIZON
                    except (TypeError, ValueError):
                        horizon_ok = False
                    if not (horizon_ok and re.search(r"\b96\b", norm_ws(quote))):
                        reason = f"Horizont nicht {ETTH1_HORIZON}"
                        emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"})
                        _update_cache(cache_key, {"status": "rejected", "reason": reason})
                        continue

                if ds in ETT_DATASETS:   # Eingabelaenge frei bis 720; nur uebernehmen, wenn die Zahl im Text steht
                    claim_data["input_length"] = _valid_input_length(claim_data.get("input_length"), text)

                method = claim_data.get("method")
                if not _method_supported(method, quote, text, _number_line(quote, target_val)):
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": "Method missing or not in quote", "stage": "quote"})
                    continue

                metric = _normalize_metric(claim_data.get("metric"))
                if metric not in ALLOWED_METRICS.get(ds, set()):
                    reason = f"Metrik nicht messbar ({claim_data.get('metric')!r})"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"})
                    _update_cache(cache_key, {"status": "rejected", "reason": reason})
                    continue
                claim_data["metric"] = metric

                # Spaltenzuordnung: steht die Zahl in der Spalte der Methode? (deterministisch, table_check.py)
                row = _number_line(quote, target_val) or quote
                col = check_columns(_table_norm(text.replace("|", " ")), _table_norm(row.replace("|", " ")), target_val,
                                    method, ds, metric, metric == "accuracy")
                if col.status == "mismatch":
                    reason = f"Number belongs to another column: {col.detail}"
                    emit({"type": "reject", "arxiv_id": cache_key, "reason": reason, "stage": "number"})
                    _update_cache(cache_key, {"status": "rejected", "reason": reason})
                    continue
                claim_data["column_check"] = "ok" if col.status == "ok" else "unchecked"

                claim_data["protocol_evidence"] = ("Protokoll: Zitat im Text belegt" if protocol == "passt"
                                                   else "Protokoll nicht belegt")
                claim = claim_data
                    
                emit({"type": "searching", "arxiv_id": cache_key, "claim": {"text": json.dumps(claim), "target": target_val}})
                    
                github_url, reason, alternatives = self._find_repo_and_context(p["title"], ds, p.get("authors", []), p["arxiv_id"])
                # Community-Hinweise (Issues, Reviews): nur als Belege, das Urteil faellt im Loop.
                community = []
                if hasattr(self.tavily, "curate_discussion"):
                    try:
                        disc = self._tavily("curate_discussion", 8, "search", p["title"], p["title"], purpose="community")
                        community = [{"url": r.get("url"), "title": r.get("title"), "snippet": (r.get("content") or "")[:300]}
                                     for r in disc.get("results", [])]
                        community = [c for c in community
                                     if _community_related(p["title"], p["arxiv_id"], c["title"], c["snippet"])]
                    except (BillingBlocked, BudgetExceeded):
                        raise
                    except Exception as e:
                        emit({"type": "warning", "arxiv_id": cache_key, "message": f"Community search failed: {e}"})

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
                    protocol=protocol,
                )
                emit(self._candidate_event(paper, cached=False))
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

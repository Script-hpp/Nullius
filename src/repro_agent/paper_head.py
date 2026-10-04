"""Paper head: plain-language summary, authors, year, affiliations and venue of a paper. Shared by scripts/make_summaries.py (stored once per paper in
web/public/data/discovery/summaries.json) and by the pipeline (a run for a new arXiv link writes the same fields into dossier.summary).

Sources, nothing else: summary = Nemotron (role code) from the arXiv abstract page and the claim; authors and year = the arXiv abstract page; affiliations
= only the paper text that Tavily Extract already read, each with a quote that must stand word for word in the head of that text; venue = only if the head
says "Published as a conference paper at <venue>". The head never goes into a prompt of the agent: it is report only.
"""

from __future__ import annotations

import html
import json
import re
from typing import Callable

import httpx

ROLE = "code"
DATASETS = {
    "Cora": "Cora, a collection of scientific articles that cite each other",
    "Cora-LCC": "Cora, a collection of scientific articles that cite each other",
    "ETTh1": "ETTh1, hourly measurements from an electricity transformer",
    "CiteSeer": "CiteSeer, a collection of scientific articles that cite each other",
    "PubMed": "PubMed, a collection of medical research articles that cite each other",
    "ETTh2": "ETTh2, hourly measurements from a second electricity transformer",
    "ETTm1": "ETTm1, measurements every 15 minutes from an electricity transformer",
    "ETTm2": "ETTm2, measurements every 15 minutes from a second electricity transformer",
}
SYSTEM = "You explain machine-learning papers to readers without any technical background. Keep reasoning brief."
HEAD_CHARS = 4500
INSTITUTION_SYSTEM = "You read the title block of a scientific paper and report the author affiliations exactly as written. Keep reasoning brief."
ACUTE = dict(zip("aeiouAEIOU", "áéíóúÁÉÍÓÚ"))
GRAVE = dict(zip("aeiouAEIOU", "àèìòùÀÈÌÒÙ"))
DEPARTMENT = re.compile(r"^(department|dept\.?|school|faculty|division|laboratory|lab)\b", re.I)


def abstract(arxiv_id: str) -> str:
    """Abstract from the arXiv abstract page (the export API often answers 429 or times out)."""
    r = httpx.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=30, follow_redirects=True)
    r.raise_for_status()
    m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', r.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", m.group(1)) if m else ""
    return html.unescape(" ".join(text.split())).removeprefix("Abstract:").strip()


def prompt(c: dict, abs_text: str) -> str:
    claim = dict(c["claim"])
    if claim["metric"] == "accuracy" and claim["target"] <= 1:
        claim["target"] = round(claim["target"] * 100, 1)   # papers.yaml stores fractions, the claim is "81.5 %"
    unit = "% accuracy" if claim["metric"] == "accuracy" else f" {claim['metric'].upper()} (lower is better)"
    return (
        f"Paper title: {c['title']}\nAbstract: {abs_text}\n\n"
        f"The claim we check: the method {claim['method']} reaches {claim['target']}{unit} on "
        f"{DATASETS.get(claim.get('matched_task') or c.get('dataset', ''), c.get('dataset', 'a benchmark'))}.\n\n"
        "Write exactly three short sentences for a reader who has never studied computer science: (1) what the paper "
        "proposes, in everyday words; (2) what the data is; (3) what the claimed number means (say what accuracy or "
        "forecast error means in plain words). Avoid jargon such as neural network, graph, oscillator, amplitude, phase, "
        "MSE, embedding; describe the idea instead. Each sentence at most 25 words. Do not judge whether the claim is true. "
        "Use only facts from the title, abstract and claim above. Style example: \"This paper proposes a new way for a "
        "computer to sort scientific articles into topics by looking at which articles cite each other. The data is "
        "Cora, a collection of about 2,700 articles. It claims the method puts 81.5% of the articles in the right topic.\" "
        'End your answer with one line of JSON: {"summary": "..."}'
    )


def parse(text: str) -> str | None:
    for m in reversed(list(re.finditer(r"\{[^{}]*\"summary\"[^{}]*\}", text, flags=re.S))):
        try:
            s = json.loads(m.group(0))["summary"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
        if 40 <= len(s) <= 700:
            return s
    return None


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s@.-]", " ", text.lower()).split())


def arxiv_meta(aid: str) -> dict:
    """Authors (First Last) and year from the arXiv abstract page meta tags."""
    r = httpx.get(f"https://arxiv.org/abs/{aid}", timeout=30, follow_redirects=True)
    r.raise_for_status()
    authors = []
    for a in re.findall(r'<meta name="citation_author" content="([^"]+)"', r.text):
        a = html.unescape(a)
        authors.append(" ".join(reversed([x.strip() for x in a.split(",", 1)])) if "," in a else a)
    m = re.search(r'<meta name="citation_date" content="(\d{4})', r.text)
    return {"authors": authors, "year": int(m.group(1)) if m else None}


def institution_prompt(head: str) -> str:
    return (
        "Below is the beginning of a paper. List the institutions in the author affiliations (universities, institutes, companies), each once. "
        "For every institution give its name exactly as it is written in the text (only the university, institute or company, without department, address or city) "
        "and a short verbatim quote from the text that contains it (copy the characters exactly, at most 160 characters). If the text names no affiliation, return an empty list. Do not guess and do not add "
        "institutions from your own knowledge.\n\n---\n" + head + "\n---\n\n"
        'End your answer with one line of JSON: {"institutions": [{"name": "...", "quote": "..."}]}'
    )


def parse_institutions(text: str) -> list[dict] | None:
    for m in reversed(list(re.finditer(r"\{\s*\"institutions\"\s*:.*\}", text, flags=re.S))):
        try:
            items = json.loads(m.group(0))["institutions"]
        except (json.JSONDecodeError, KeyError):
            continue
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict) and isinstance(i.get("name"), str) and isinstance(i.get("quote"), str)]
    return None


def clean_name(name: str) -> str:
    """Display name: undo PDF text artifacts (accent marks apart from their letter, hyphenation across lines). The quote stays verbatim."""
    name = re.sub(r"´ ?([aeiouAEIOU])", lambda m: ACUTE[m.group(1)], name)
    name = re.sub(r"` ?([aeiouAEIOU])", lambda m: GRAVE[m.group(1)], name)
    name = re.sub(r"(?<=[a-z])- (?=[a-z])|(?<=[a-z])-(?=[a-z]{3})", "", name) if re.search(r"[a-z]-[a-z]", name) and not re.search(r"[A-Z][a-z]+-[A-Z]", name) else name
    return " ".join(name.split())


def drop_departments(items: list[dict]) -> list[dict]:
    """A department or school is dropped when the paper also names a university, institute or company (the quote of a department often
    ends at a line break, so the university sits in its own entry)."""
    if not any(not DEPARTMENT.match(i["name"].strip()) for i in items):
        return items
    return [i for i in items if not DEPARTMENT.match(i["name"].strip())]


def verified_institutions(items: list[dict], head: str) -> list[dict]:
    """Keep an institution only if its quote stands word for word in the head of the text and its name stands in the quote."""
    h = norm(head)
    out, seen = [], set()
    for i in items:
        q, n = norm(i["quote"]), norm(i["name"])
        if len(n) < 3 or len(q) < len(n) or q not in h or n not in q or n in seen:
            continue
        seen.add(n)
        out.append({"name": clean_name(i["name"].strip()), "quote": i["quote"].strip()})
    return drop_departments(out)


def venue_of(head: str) -> dict | None:
    m = re.search(r"(Published as a conference paper at ([A-Z][A-Za-z]+(?: [A-Za-z]+)? ?\d{4}))", head)
    return {"name": m.group(2).strip(), "quote": m.group(1)} if m else None


def source_label(fix: dict | None) -> str | None:
    if not fix:
        return None
    kind = {"arxiv_html": "HTML", "ar5iv": "HTML (ar5iv)", "pdf": "PDF"}.get(fix.get("source_kind") or "", "text")
    return " ".join(x for x in ["arXiv", kind, fix.get("arxiv_version") or ""] if x) + ", read with Tavily Extract"


def head_of_text(text: str) -> str:
    """The part of a paper text that holds the title block. HTML texts start with page navigation: begin shortly before the first "Affiliation" or at
    the title line; PDF texts start with the title block."""
    a = text.find("Affiliation")
    t = re.search(r"^# .+$", text, flags=re.M)
    start = max(0, a - 900) if a >= 0 else (max(0, t.start() - 120) if t else 0)   # a few characters before the title: PDFs put the venue line there
    return text[start:start + HEAD_CHARS]


def claim_for_prompt(paper: dict, claim) -> dict:
    """The input of prompt(): title, dataset and the claim as {method, metric, target}. The method is the one the paper's fidelity entry names, else the title."""
    method = (paper.get("fidelity") or {}).get("method") or paper.get("claim_method") or paper["title"]
    return {"title": paper["title"], "dataset": paper.get("dataset", ""), "claim": {"method": method, "metric": claim.metric, "target": claim.target}}


def build_head(ask: Callable[[str, str, float], str], c: dict, abstract_text: str, meta: dict | None, doc: dict | None, arxiv_id: str) -> dict:
    """One paper head. ask(system, user, temperature) -> answer text (the caller decides model, retries and cost). `meta` = arxiv_meta() result or None,
    `doc` = the paper text Tavily already read ({"text", "source_kind", "arxiv_version"}) or None. Raises what ask raises; the caller keeps the run going.
    Result: {"text", "source", "generated_by", "authors", "authors_source", "year", "institutions", "institutions_source", "venue"}, keys without data left out."""
    out: dict = {}
    p = prompt(c, abstract_text)
    for t in (0.2, 0.5):
        summary = parse(ask(SYSTEM, p, t) or "")
        if summary:
            out.update({"text": summary, "source": f"arXiv abstract ({arxiv_id}) and the claim of the check", "generated_by": "Nemotron Super (Token Factory)"})
            break
    if meta and meta.get("authors"):
        out.update({"authors": meta["authors"], "year": meta.get("year"), "authors_source": f"arXiv abstract page ({arxiv_id})"})
    if doc and doc.get("text"):
        head = head_of_text(doc["text"])
        items = parse_institutions(ask(INSTITUTION_SYSTEM, institution_prompt(head), 0.1) or "")
        kept = verified_institutions(items or [], head)
        out.update({"institutions": kept, "institutions_source": source_label(doc)})
        v = venue_of(head)
        if v:
            out["venue"] = v
    return out

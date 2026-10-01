"""Paper heads (web/public/data/discovery/summaries.json): plain-language summary, authors, affiliations, year, venue. Keyed by arXiv id.

Covers every paper the app shows: benchmark papers (benchmark/papers.yaml), catalog entries (benchmark/catalog.yaml) and Discovery candidates.
Generated once and stored; the web app never calls a model per page view. Sources, nothing else:
- summary: Nemotron (role code) from the arXiv abstract page and the claim (as before); an existing summary is kept unless --force.
- authors, year: the arXiv abstract page (citation_author / citation_date meta tags).
- institutions: only from the paper text that Tavily Extract already stored (benchmark/fixtures/<arxiv_id>.json, NO new Tavily call). Nemotron
  returns names with a quote; an entry is kept only if the quote stands word for word in the head of that text and the name stands in the quote.
  Papers without a stored text get no institutions.
- venue: only if the head of the text says "Published as a conference paper at <venue>".

Usage: LLM_PROVIDER=tokenfactory .venv/bin/python scripts/make_summaries.py [--force] [--dry] [--redo-institutions] [--redo-summary=id,id]
(--dry: no model call, shows what would be asked)
"""

import html
import json
import re
import sys
from pathlib import Path

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from repro_agent.llm import LLMClient  # noqa: E402

FIXTURES = REPO_ROOT / "benchmark" / "fixtures"
DATA = REPO_ROOT / "web" / "public" / "data" / "discovery"
OUT = DATA / "summaries.json"
ROLE = "code"   # Nemotron Super; the 'fast' model writes its reasoning into the answer
DATASETS = {
    "Cora": "Cora, a collection of scientific articles that cite each other",
    "Cora-LCC": "Cora, a collection of scientific articles that cite each other",
    "ETTh1": "ETTh1, hourly measurements from an electricity transformer",
}
SYSTEM = "You explain machine-learning papers to readers without any technical background. Keep reasoning brief."


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


HEAD_CHARS = 4500
INSTITUTION_SYSTEM = "You read the title block of a scientific paper and report the author affiliations exactly as written. Keep reasoning brief."


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


def fixture_head(aid: str) -> tuple[str, dict] | None:
    f = FIXTURES / f"{aid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    text = d.get("text") or ""
    if not text:
        return None
    # HTML texts start with page navigation: begin shortly before the first "Affiliation" or at the title line; PDF texts start with the title block
    a = text.find("Affiliation")
    t = re.search(r"^# .+$", text, flags=re.M)
    start = max(0, a - 900) if a >= 0 else (t.start() if t else 0)
    return text[start:start + HEAD_CHARS], d


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


ACUTE = dict(zip("aeiouAEIOU", "áéíóúÁÉÍÓÚ"))
GRAVE = dict(zip("aeiouAEIOU", "àèìòùÀÈÌÒÙ"))


def clean_name(name: str) -> str:
    """Display name: undo PDF text artifacts (accent marks apart from their letter, hyphenation across lines). The quote stays verbatim."""
    name = re.sub(r"´ ?([aeiouAEIOU])", lambda m: ACUTE[m.group(1)], name)
    name = re.sub(r"` ?([aeiouAEIOU])", lambda m: GRAVE[m.group(1)], name)
    name = re.sub(r"(?<=[a-z])- (?=[a-z])|(?<=[a-z])-(?=[a-z]{3})", "", name) if re.search(r"[a-z]-[a-z]", name) and not re.search(r"[A-Z][a-z]+-[A-Z]", name) else name
    return " ".join(name.split())


DEPARTMENT = re.compile(r"^(department|dept\.?|school|faculty|division|laboratory|lab)\b", re.I)


def drop_departments(items: list[dict]) -> list[dict]:
    """A department or school is dropped when the paper also names a university, institute or company (the quote of a department often
    ends at a line break, so the university sits in its own entry)."""
    if not any(not DEPARTMENT.match(i["name"].strip()) for i in items):
        return items
    return [i for i in items if not DEPARTMENT.match(i["name"].strip())]


def venue_of(head: str) -> dict | None:
    m = re.search(r"(Published as a conference paper at ([A-Z][A-Za-z]+(?: [A-Za-z]+)? ?\d{4}))", head)
    return {"name": m.group(2).strip(), "quote": m.group(1)} if m else None


def source_label(fix: dict | None) -> str | None:
    if not fix:
        return None
    kind = {"arxiv_html": "HTML", "pdf": "PDF"}.get(fix.get("source_kind") or "", "text")
    return " ".join(x for x in ["arXiv", kind, fix.get("arxiv_version") or ""] if x) + ", read with Tavily Extract"


def collect() -> dict[str, dict]:
    """arXiv id -> {title, paper_ids, claim, dataset}: benchmark papers, catalog entries, Discovery candidates."""
    papers: dict[str, dict] = {}
    for p in yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]:
        m = re.search(r"arxiv\.org/abs/([\d.]+)", p.get("arxiv_url") or "")
        if not m:
            continue
        claim = next((c for c in p.get("claims", []) if "placebo" not in c["id"]), None)
        e = papers.setdefault(m.group(1), {"title": p["title"], "paper_ids": [], "dataset": p.get("dataset", "")})
        e["paper_ids"].append(p["id"])
        if claim:
            e["claim"] = {"method": (p.get("fidelity") or {}).get("method", p["title"]), "metric": claim["metric"], "target": claim["target"]}
    for c in yaml.safe_load((REPO_ROOT / "benchmark" / "catalog.yaml").read_text()).get("candidates", []):
        cand = c.get("candidate") or {}
        e = papers.setdefault(c["arxiv_id"], {"title": c["title"], "paper_ids": [], "dataset": cand.get("dataset", "")})
        e["paper_ids"].append(c["id"])
        if cand.get("claim") and "claim" not in e:
            e["claim"] = cand["claim"]
    replay = DATA / "replay.jsonl"
    for x in (replay.read_text().splitlines() if replay.exists() else []):
        c = json.loads(x) if x.strip() else {}
        if c.get("type") == "candidate" and c.get("arxiv_id"):
            e = papers.setdefault(c["arxiv_id"], {"title": c["title"], "paper_ids": [], "dataset": c.get("dataset", "")})
            e.setdefault("claim", c.get("claim"))
    return papers


def main() -> None:
    force, dry = "--force" in sys.argv, "--dry" in sys.argv
    redo_inst = "--redo-institutions" in sys.argv
    if "--refilter" in sys.argv:   # no model call: apply the department filter to the stored institutions
        out = json.loads(OUT.read_text())
        for aid, e in out.items():
            before = [i["name"] for i in e.get("institutions", [])]
            e["institutions"] = drop_departments(e.get("institutions", [])) if "institutions" in e else e.get("institutions")
            if "institutions" not in e or e["institutions"] is None:
                e.pop("institutions", None)
            elif [i["name"] for i in e["institutions"]] != before:
                print(aid, before, "->", [i["name"] for i in e["institutions"]])
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        return
    redo_sum = set(next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--redo-summary=")), []))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    papers = collect()
    llm = None if dry else LLMClient()
    for aid, c in papers.items():
        entry = out.get(aid, {})
        # 1. summary (kept unless --force or a hand-written example)
        if (force or aid in redo_sum or not entry.get("summary") or entry.get("example")) and c.get("claim"):
            cc = {"title": c["title"], "dataset": c.get("dataset", ""), "claim": c["claim"]}
            if dry:
                print(f"{aid}: would write a summary")
            else:
                p = prompt(cc, abstract(aid))
                summary = None
                for t in (0.2, 0.5):
                    resp = llm.complete(ROLE, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}], temperature=t, max_tokens=3000)
                    summary = parse(resp.content or "")
                    if summary:
                        break
                if summary:
                    entry.update({"summary": summary, "source": f"arXiv abstract ({aid}) and the claim of the check", "generated_by": "Nemotron Super (Token Factory)"})
                    entry.pop("example", None)
                    print(f"{aid}: {summary}")
                else:
                    print(f"{aid}: no valid summary")
        # 2. authors and year (arXiv abstract page)
        if force or "authors" not in entry:
            if dry:
                print(f"{aid}: would read authors from arxiv.org/abs")
            else:
                meta = arxiv_meta(aid)
                entry.update({"authors": meta["authors"], "year": meta["year"], "authors_source": f"arXiv abstract page ({aid})"})
        # 3. institutions and venue (stored paper text, no Tavily call)
        fh = fixture_head(aid)
        if fh and (force or redo_inst or "institutions" not in entry):
            head, fix = fh
            if dry:
                print(f"{aid}: would ask Nemotron for affiliations from {len(head)} characters of the stored text")
            else:
                resp = llm.complete(ROLE, [{"role": "system", "content": INSTITUTION_SYSTEM}, {"role": "user", "content": institution_prompt(head)}], temperature=0.1, max_tokens=3000)
                items = parse_institutions(resp.content or "")
                kept = verified_institutions(items or [], head)
                print(f"{aid}: {len(kept)} of {len(items or [])} affiliations verified: {[k['name'] for k in kept]}")
                entry.update({"institutions": kept, "institutions_source": source_label(fix)})
                v = venue_of(head)
                if v:
                    entry["venue"] = v
        elif not fh and "institutions" not in entry:
            print(f"{aid}: no stored paper text, no institutions")
        if entry:
            entry["paper_ids"] = c["paper_ids"]
            entry["title"] = c["title"]
            out[aid] = entry
    if llm is not None:
        print(f"cost of this run: {llm.total_cost_usd:.4f} USD net, {llm.gross(llm.total_cost_usd):.4f} USD gross")
    if not dry:
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

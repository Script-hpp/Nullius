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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from repro_agent.llm import LLMClient  # noqa: E402
from repro_agent.paper_head import (  # noqa: E402
    INSTITUTION_SYSTEM, ROLE, SYSTEM, abstract, arxiv_meta, head_of_text, institution_prompt, parse, parse_institutions, prompt,
    source_label, venue_of, verified_institutions,
)

FIXTURES = REPO_ROOT / "benchmark" / "fixtures"
DATA = REPO_ROOT / "web" / "public" / "data" / "discovery"
OUT = DATA / "summaries.json"




def fixture_head(aid: str) -> tuple[str, dict] | None:
    f = FIXTURES / f"{aid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return (head_of_text(d["text"]), d) if d.get("text") else None


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

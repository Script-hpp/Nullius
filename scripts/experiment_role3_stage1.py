"""Rolle 3, Stufe 1 (Regeln: benchmark/experiments/role3_dlinear.md). Ein Tavily-Aufruf (Search + Crawl mit limit),
Ergebnis als Fixture; Extraktion mit Rolle `extract`; Abgleich mit dem Repo ohne LLM.

  LLM_PROVIDER=tokenfactory uv run python scripts/experiment_role3_stage1.py
"""

from __future__ import annotations

import dataclasses
import json

from repro_agent.claims_source import fetch_paper_text
from repro_agent.dossier import REPO_ROOT
from repro_agent.llm import LLMClient
from repro_agent.objections import docs_from_collect, extract_objections
from repro_agent.repo_source import find_repo_url

TITLE = "Are Transformers Effective for Time Series Forecasting?"
FIXTURE = REPO_ROOT / "benchmark" / "fixtures" / "role3_zeng_objections.json"
REPO_FIXTURE = REPO_ROOT / "benchmark" / "fixtures" / "repos" / "cure-lab_LTSF-Linear.json"


def main() -> None:
    text = fetch_paper_text("2205.13504", lambda: (_ for _ in ()).throw(RuntimeError("Fixture fehlt"))).text
    repo_url = find_repo_url(text)
    if FIXTURE.exists():
        collected = json.loads(FIXTURE.read_text())
        print("Fixture vorhanden, kein Tavily-Aufruf")
    else:
        from repro_agent.tavily_tools import TavilyTools

        collected = TavilyTools().collect_objections(TITLE, repo_url=repo_url, limit=5)   # der EINE Aufruf
        FIXTURE.write_text(json.dumps(collected, ensure_ascii=False, indent=1))
    search = (collected.get("search") or {}).get("results", [])
    crawl = collected.get("crawl") or {}
    print(f"Repo {repo_url}; Search {len(search)} Treffer, Crawl {len(crawl.get('results', []))} Seiten"
          + (f" (Fehler: {crawl['error']})" if crawl.get("error") else ""))
    for r in search:
        print("  search:", r.get("url"), len(r.get("raw_content") or ""))
    for r in crawl.get("results", []):
        print("  crawl:", r.get("url"), len(r.get("raw_content") or ""))
    docs = docs_from_collect(collected, ("dlinear", "ltsf"))
    print(f"{len(docs)} Seiten nennen DLinear/LTSF")
    repo_files = json.loads(REPO_FIXTURE.read_text())["files"]
    llm = LLMClient()
    objs, dropped = extract_objections(llm, docs, TITLE, "ETTh1", repo_files)
    print(f"{len(objs)} verifizierte Einwaende, {dropped} verworfen, Kosten {llm.total_cost_usd:.4f} $")
    for o in objs:
        print(json.dumps(dataclasses.asdict(o), ensure_ascii=False))


if __name__ == "__main__":
    main()

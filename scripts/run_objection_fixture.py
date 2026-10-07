"""Probelauf der Einwand-Experimente auf gespeicherten Community-Seiten (kein Tavily-Aufruf).

Echte Token-Factory- und Sandbox-Aufrufe (LLM_PROVIDER=tokenfactory), aber die Funde kommen aus benchmark/fixtures/role3_zeng_objections.json
statt aus einer Live-Suche. Ablauf wie ein Produktlauf (graph_repair=True), Tavily-Rollen 2/4 aus Fixtures; Community-Funde aus der Fixture,
danach die Einwand-Experimente. Vorab festgelegt in benchmark/experiments/tavily_experiments_test.md.
  LLM_PROVIDER=tokenfactory uv run python scripts/run_objection_fixture.py [--paper zeng_dlinear_2023] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent import community, objection_experiments  # noqa: E402
from repro_agent.loop import Orchestrator  # noqa: E402

FIXTURE = ROOT / "benchmark" / "fixtures" / "role3_zeng_objections.json"
ARXIV = {"zeng_dlinear_2023": "2205.13504"}


class FixtureTools:
    """Nur die Community-Methoden, alle aus der gespeicherten Fixture; jeder andere Aufruf ist ein Fehler."""

    def __init__(self) -> None:
        self.data = json.loads(FIXTURE.read_text())

    def search_community(self, title, limit=5):
        return self.data["search"]

    def search_method_critique(self, method, dataset, limit=5):
        return {"results": []}

    def search_repo_issues(self, slug, limit=5):
        return {"results": []}

    def crawl_issues(self, repo_url, limit=3):
        return self.data["crawl"]

    def read_pages(self, urls, limit=3):
        raise AssertionError("fixture run must not read pages")

    def clarify_discrepancy(self, title, max_results=3):
        raise AssertionError("fixture run must not search explanations")


class FixtureOrchestrator(Orchestrator):
    def _community_round(self, paper, dossier):
        title, dataset = paper["title"], paper["dataset"]
        spec = ((paper.get("claims") or [{}])[0].get("extract") or {})
        found = community.what_others_found(FixtureTools(), self.llm, title, ARXIV[dossier.paper_id], spec.get("dataset") or dataset,
                                            (dossier.repo or {}).get("url"), getattr(self, "_repo_files", None), None, method=spec.get("method"))
        dossier.community = dict(found, mode="fixture", queried_at="fixture")
        self._emit("community", mode="fixture", n_findings=len(found["findings"]), dropped=found["dropped"], queries=found["queries"],
                   reused_discovery=False, findings=found["findings"])

    def _objection_round(self, paper, dossier):
        objection_experiments.run_round(self, dossier, paper, quick=self.objection_quick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="zeng_dlinear_2023")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    orch = FixtureOrchestrator(graph_repair=True, tavily_mode="fixture", llm_cache=False, objection_experiments=True,
                               objection_quick=a.quick, seeds=a.seeds, claim_ids=["dlinear_etth1_96_mse"])
    dossier = orch.run(a.paper)
    print("run dir:", orch.last_run_dir)
    for c in dossier.objection_checks:
        print(c["id"], c["type"], c["status"], "|", (c.get("note") or "")[:80], "|", c["quote"][:70])


if __name__ == "__main__":
    main()

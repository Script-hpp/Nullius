"""Bausteinliste einer Methode aus dem Paper extrahieren (einmal), zur Handpruefung ablegen.

  LLM_PROVIDER=tokenfactory uv run python scripts/extract_components.py <paper_id>

Ergebnis: benchmark/components/<paper_id>.json mit reviewed=false. Nach der Durchsicht von Hand
reviewed auf true setzen; die Liste wird dann fuer alle Laeufe und Arme eingefroren.
"""

from __future__ import annotations

import sys
import time

import yaml

from repro_agent.claims_source import fetch_paper_text
from repro_agent.dossier import REPO_ROOT
from repro_agent.fidelity import extract_components, save_components
from repro_agent.llm import LLMClient


def main() -> None:
    paper_id = sys.argv[1]
    papers = yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
    paper = next(p for p in papers if p["id"] == paper_id)
    arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
    text = fetch_paper_text(arxiv_id, lambda: (_ for _ in ()).throw(RuntimeError("Fixture fehlt")))
    llm = LLMClient()
    spec = paper["fidelity"]
    comps, dropped = extract_components(llm, text, spec["method"], spec["keywords"])
    meta = {"paper_id": paper_id, "source": text.url, "sha256": text.sha256, "dropped": len(dropped),
            "extracted_at": time.strftime("%Y-%m-%d"), "cost_usd": round(llm.total_cost_usd, 4)}
    path = save_components(paper_id, comps, meta)
    print(f"{len(comps)} Bausteine, {len(dropped)} verworfen -> {path}")
    for d in dropped:
        print("  verworfen:", d)
    for c in comps:
        print(f"- [{'kritisch' if c.critical else 'klein'}] {c.id}: {c.name} | Wert: {c.value}")


if __name__ == "__main__":
    main()

"""Seeded discovery: send given arXiv ids through the normal discovery pipeline.

  uv run python scripts/discover_ids.py --ids 2405.00946,2307.03756 --dataset ETTh1

Same checks as search discovery (Tavily extract of the paper text, claim with verbatim quote, protocol check,
feasibility rule, verified repo search). Only the search step is replaced by the given ids. Candidates land in
logs/discovery_cache.json with found_by="seeded". Ids already in the benchmark or catalog are refused.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ("ETTh1", "Cora", "CiteSeer")


def known_ids(root: Path = ROOT) -> dict[str, str]:
    """arXiv ids already in benchmark/papers.yaml or benchmark/catalog.yaml -> where."""
    found: dict[str, str] = {}
    pat = re.compile(r"(\d{4}\.\d{4,5})")
    papers = root / "benchmark" / "papers.yaml"
    if papers.exists():
        for m in pat.finditer(papers.read_text()):
            found.setdefault(m.group(1), "benchmark/papers.yaml")
    catalog = root / "benchmark" / "catalog.yaml"
    if catalog.exists():
        for e in yaml.safe_load(catalog.read_text()) or []:
            if isinstance(e, dict) and e.get("arxiv_id"):
                found.setdefault(str(e["arxiv_id"]), "benchmark/catalog.yaml")
    return found


def split_ids(raw: str, known: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    ids = list(dict.fromkeys(re.sub(r"v\d+$", "", i.strip()) for i in raw.split(",") if i.strip()))
    refused = {i: known[i] for i in ids if i in known}
    return [i for i in ids if i not in refused], refused


def main(argv: list[str] | None = None, agent=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="comma-separated arXiv ids")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--allow-known", action="store_true", help="do not refuse ids already in benchmark/catalog")
    args = ap.parse_args(argv)

    known = {} if args.allow_known else known_ids()
    ids, refused = split_ids(args.ids, known)
    for i, where in refused.items():
        print(f"SKIP {i}: already in {where}")
    if not ids:
        print("nothing to do")
        return 0
    if agent is None:
        from repro_agent.discovery import DiscoveryAgent
        agent = DiscoveryAgent()
    events: list[dict] = []
    cands = agent.discover([args.dataset], seed_ids=ids, on_event=events.append)
    for e in events:
        if e.get("type") == "reject":
            print(f"REJECT {e.get('arxiv_id')}: [{e.get('stage')}] {e.get('reason')}")
        elif e.get("type") in ("error", "warning"):
            print(f"{e['type'].upper()}: {e.get('message')}")
        elif e.get("type") == "candidate":
            print(f"CANDIDATE {e['arxiv_id']}: {e['title'][:70]} | protocol={e['protocol']} | repo={e.get('github_url')} ({e.get('repo_status')})")
    print(json.dumps({"requested": ids, "candidates": [c.arxiv_id for c in cands]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Misst, wie viele Rolle-4-Angaben die Zitatpruefung verwirft und warum, auf der Entwicklungsmenge (nicht GAT, nicht DGI).

  LLM_PROVIDER=tokenfactory uv run python scripts/measure_role4_drops.py collect   # einmal: Rohangaben des Modells sammeln (LLM)
  uv run python scripts/measure_role4_drops.py evaluate [alt|neu]                   # deterministisch, ohne LLM, auf denselben Rohangaben
Die Rohangaben werden gespeichert (benchmark/fixtures/role4_raw_<paper>.json), damit Vorher/Nachher dieselben Modellausgaben sehen.
"""

from __future__ import annotations

import json
import sys

import yaml

from repro_agent import claims_source
from repro_agent.dossier import REPO_ROOT
from repro_agent.repo_source import FIXTURE_DIR as REPO_FIX, _repo_key, find_repo_url
from repro_agent.setup_details import Doc, collect_raw, verify_raw

PAPERS = {"kipf_gcn_2017": "1609.02907", "wu_sgc_2019": "1902.07153", "shchur_pitfalls_2018": "1811.05868", "zeng_dlinear_2023": "2205.13504"}
FIX = REPO_ROOT / "benchmark" / "fixtures"


def docs_for(paper_id: str) -> list[Doc]:
    text = json.loads((FIX / f"{PAPERS[paper_id]}.json").read_text())["text"]
    docs = [Doc("paper", text)]
    url = find_repo_url(text)
    if url and (REPO_FIX / f"{_repo_key(url)}_readme.json").exists():
        docs.append(Doc("repo-readme", json.loads((REPO_FIX / f"{_repo_key(url)}_readme.json").read_text())["text"]))
    if url and (REPO_FIX / f"{_repo_key(url)}.json").exists():
        dump = json.loads((REPO_FIX / f"{_repo_key(url)}.json").read_text())
        docs += [Doc(f"repo:{p}@{dump['commit'][:7]}", t) for p, t in dump["files"].items()]
    return docs


def main() -> None:
    papers = yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
    if sys.argv[1] == "collect":
        from repro_agent.config import load_settings
        from repro_agent.llm import LLMClient
        import dataclasses

        llm = LLMClient(dataclasses.replace(load_settings(), llm_cache=False))
        for pid in PAPERS:
            p = next(x for x in papers if x["id"] == pid)
            raw = collect_raw(llm, docs_for(pid), p["claims"][0]["text"], p["dataset"])
            (FIX / f"role4_raw_{pid}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1))
            print(pid, len(raw), "Rohangaben", f"{llm.total_cost_usd:.4f} $ bisher", flush=True)
        return
    which = sys.argv[2] if len(sys.argv) > 2 else "alt"
    alt = which == "alt"
    norm = claims_source._norm_v1 if alt else claims_source._norm
    tot_kept = tot_drop = 0
    from collections import Counter
    reasons: Counter = Counter()
    for pid in PAPERS:
        raw = json.loads((FIX / f"role4_raw_{pid}.json").read_text())
        kept, dropped = verify_raw(raw, docs_for(pid), norm, elisions=not alt, check_numbers=not alt)
        tot_kept += len(kept)
        tot_drop += len(dropped)
        reasons.update(d["reason"] for d in dropped)
        print(f"{pid}: {len(raw)} Rohangaben, {len(kept)} behalten, {len(dropped)} verworfen ({len(dropped) / max(1, len(raw)):.0%})")
        if "-v" in sys.argv:
            for d in dropped[:60]:
                it = d["item"] if isinstance(d["item"], dict) else {}
                print("   ", d["reason"], "|", d["doc"][:24], "|", str(it.get("topic"))[:24], "|", repr(str(it.get("quote"))[:110]))
    print(f"Summe ({which}): {tot_kept} behalten, {tot_drop} verworfen ({tot_drop / max(1, tot_kept + tot_drop):.0%}); Gruende: {dict(reasons)}")


if __name__ == "__main__":
    main()

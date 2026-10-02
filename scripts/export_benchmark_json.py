"""Write web/public/data/benchmark.json from the benchmark stores (v4, v3, v2; newest first).

Same counting rule as scripts/eval_benchmark_v2.py (majority per paper-claim unit over 3 generations, axis 1 only),
same reference verdicts (benchmark/experiments/benchmark_v2.md). No model calls. Format: web/src/benchmark.ts.
Usage: .venv/bin/python scripts/export_benchmark_json.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_benchmark_v2 as ev  # noqa: E402

OUT = REPO_ROOT / "web" / "public" / "data" / "benchmark.json"
# (id, store); newest first, the landing page shows the first entry
VERSIONS = [
    ("v4", "runs/benchmark_v4.jsonl"),
    ("v3", "runs/benchmark_v3.jsonl"),
    ("v2", "runs/benchmark_v2.jsonl"),
]


def _iso(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d


def version(vid: str, rows: list[dict], refs: dict, papers: dict) -> dict:
    ref = ev.reference_verdicts(refs)
    units, agree, placebo_passed = [], Counter(), 0
    for unit, r in ref.items():
        arms, placebo = {}, {}
        for key, arm in (("A", ev.ARMS[0]), ("B", ev.ARMS[1])):
            mine = sorted((x for x in rows if x["paper"] == unit and x["arm"] == arm), key=lambda x: x["generation"])
            real = [x["urteil"] for x in mine if x["claim"] == r["claim"]]
            plac = [x["urteil"] for x in mine if x["claim"] == r["placebo"]]
            maj = ev.majority(real)
            arms[key] = {"majority": maj, "verdicts": real}
            placebo[key] = plac
            agree[key] += maj == r["ref"]
            placebo_passed += sum(u == "reproduziert" for u in plac)
        claim = next(c for c in papers[unit]["claims"] if c["id"] == r["claim"])
        units.append({
            "paper": unit, "title": papers[unit]["title"], "claim": r["claim"], "metric": claim["metric"],
            "target": r["target"], "reference_verdict": r["ref"], "arms": arms, "placebo": placebo,
        })
    commits = Counter(x.get("code_commit", "") for x in rows)
    return {
        "id": vid, "date": _iso(max((x["run_dir"].rsplit("/", 1)[-1][:8] for x in rows), default="")),
        "code_commit": ", ".join(c for c, _ in commits.most_common()), "n_runs": len({x["run_dir"] for x in rows}),
        "units": units,
        "summary": {"agree_A": agree["A"], "agree_B": agree["B"], "placebo_passed": placebo_passed, "n_units": len(units)},
    }


def main() -> None:
    refs = ev.load_refs()
    papers = {p["id"]: p for p in yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    for p in papers.values():   # claims as a list, as in eval_benchmark_v2
        if isinstance(p.get("claims"), dict):
            p["claims"] = [p["claims"]]
    out = []
    for vid, store in VERSIONS:
        path = REPO_ROOT / store
        if path.exists():
            out.append(version(vid, [json.loads(x) for x in path.read_text().splitlines()], refs, papers))
    OUT.write_text(json.dumps({"versions": out}, ensure_ascii=False, indent=1) + "\n")
    for v in out:
        print(v["id"], v["n_runs"], "runs", v["summary"])


if __name__ == "__main__":
    main()

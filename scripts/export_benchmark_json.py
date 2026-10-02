"""Write web/public/data/benchmark.json from the benchmark stores (v4, v3, v2; newest first).

Same counting rule as scripts/eval_benchmark_v2.py (majority per paper-claim unit over 3 generations, axis 1 only),
same reference verdicts (benchmark/experiments/benchmark_v2.md). No model calls. Format: web/src/benchmark.ts.
Usage: .venv/bin/python scripts/export_benchmark_json.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_benchmark_v2 as ev  # noqa: E402

RUNS_DIR = Path(os.environ.get("BENCHMARK_RUNS_DIR", REPO_ROOT / "runs"))   # read only; runs/ is not in git
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
    units, agree, placebo_passed, placebo_total = [], Counter(), 0, 0
    real_all, real_by_arm = [], {"A": [], "B": []}   # post hoc: single "reproduziert" verdicts of real claims and distances to the target
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
            placebo_total += len(plac)
            real_all += [(u, r["ref"]) for u in real]
            real_by_arm[key] += [x for x in mine if x["claim"] == r["claim"]]
        claim = next(c for c in papers[unit]["claims"] if c["id"] == r["claim"])
        units.append({
            "paper": unit, "title": papers[unit]["title"], "claim": r["claim"], "metric": claim["metric"],
            "target": r["target"], "reference_verdict": r["ref"], "arms": arms, "placebo": placebo,
        })
    summary = {"agree_A": agree["A"], "agree_B": agree["B"], "placebo_passed": placebo_passed, "n_units": len(units),
               "placebo_reproduced": placebo_passed, "placebo_total": placebo_total,
               "reproduced_real": sum(u == "reproduziert" for u, _ in real_all),
               "reproduced_real_agree": sum(u == "reproduziert" and ref_v == "reproduziert" for u, ref_v in real_all)}
    for key in ("A", "B"):
        dist = ev.distances(real_by_arm[key])
        if dist:   # no estimate when the store has no measured means
            summary[f"median_distance_{key}"] = statistics.median(dist)
            summary[f"median_distance_n_{key}"] = len(dist)
    commits = Counter(x.get("code_commit", "") for x in rows)
    return {
        "id": vid, "date": _iso(max((x["run_dir"].rsplit("/", 1)[-1][:8] for x in rows), default="")),
        "code_commit": ", ".join(c for c, _ in commits.most_common()), "n_runs": len({x["run_dir"] for x in rows}),
        "units": units,
        "summary": summary,
    }


def main() -> None:
    refs = ev.load_refs()
    papers = {p["id"]: p for p in yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    for p in papers.values():   # claims as a list, as in eval_benchmark_v2
        if isinstance(p.get("claims"), dict):
            p["claims"] = [p["claims"]]
    out = []
    for vid, store in VERSIONS:
        path = RUNS_DIR / Path(store).name
        if path.exists():
            out.append(version(vid, [json.loads(x) for x in path.read_text().splitlines()], refs, papers))
    OUT.write_text(json.dumps({"versions": out}, ensure_ascii=False, indent=1) + "\n")
    for v in out:
        print(v["id"], v["n_runs"], "runs", v["summary"])


if __name__ == "__main__":
    main()

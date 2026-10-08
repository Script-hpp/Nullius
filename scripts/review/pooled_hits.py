"""Pooled per-generation hit rates per unit and arm over all benchmark stores (read only, stdlib).

Usage: python3 scripts/review/pooled_hits.py <main checkout>
A generation "hits" when its Axis-1 class equals the unit's reference verdict (benchmark_v2.md).
Majority probability uses p^3 + 3 p^2 (1 - p), generations treated as independent. Pooling mixes
code versions (v2 d71a8af, v3 4bab96e, v4 bf520c1, v5 C 9557def): it describes the system family, not one commit.
"""
import collections, json, math, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
REF = {"kipf_gcn_2017": (0.815, "rep"), "wu_sgc_2019": (0.810, "rep"), "zeng_dlinear_2023": (0.375, "rep"),
       "shchur_pitfalls_2018": (0.815, "rep"), "velickovic_gat_2018": (0.830, "rep"),
       "velickovic_dgi_2019": (0.823, "teil"), "shchur_gat_2018": (0.818, "rep"), "zeng_nlinear_2023": (0.374, "rep")}
STORES = ["runs/benchmark_v2.jsonl", "runs/benchmark_v3.jsonl", "runs/benchmark_v4.jsonl", "runs/v5_step_c.jsonl"]
rows = collections.defaultdict(list)
for s in STORES:
    for line in open(ROOT / s):
        r = json.loads(line)
        if r["origin"] == "paper":
            m = r["mean"]
            rows[(r["paper"], r["arm"][0])].append(None if m is None or (isinstance(m, float) and math.isnan(m)) else m)
total = collections.Counter()
for p, (t, cls) in REF.items():
    for arm in "AB":
        v = rows[(p, arm)]
        hits = sum(1 for m in v if m is not None and ((0.01 < abs(m - t) <= 0.03) if cls == "teil" else abs(m - t) <= 0.01))
        q = hits / len(v)
        maj = q ** 3 + 3 * q * q * (1 - q)
        total[arm] += maj
        print(f"{p:22} {arm} hits {hits}/{len(v)} p={q:.2f} P(majority)={maj:.2f} failures={sum(m is None for m in v)}")
print({a: round(x, 2) for a, x in total.items()}, "expected matching units of 8")

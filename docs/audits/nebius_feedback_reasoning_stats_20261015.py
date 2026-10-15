"""Reasoning-token share and finish_reason=length counts per model from a local logs/costs.jsonl.
Usage: python -I docs/audits/nebius_feedback_reasoning_stats_20261015.py <path-to-costs.jsonl> [...]
Filter (exact): provider == "tokenfactory", cache_hit false, field reasoning_tokens present.
Reads only the local log; no network, no API calls."""
import collections
import datetime as d
import json
import sys

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    f = [r for r in rows if r.get("provider") == "tokenfactory" and not r.get("cache_hit") and "reasoning_tokens" in r]
    iso = lambda t: d.datetime.fromtimestamp(t, d.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{path}: rows={len(rows)} filtered={len(f)} {iso(min(r['ts'] for r in f))} .. {iso(max(r['ts'] for r in f))} UTC")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in f:
        a = agg[r["model"]]
        a[0] += 1
        a[1] += r["completion_tokens"]
        a[2] += r["reasoning_tokens"]
        a[3] += r.get("finish_reason") == "length"
        a[4] += r.get("finish_reason") == "length" and r.get("status") == "bad_response"
    for m, a in sorted(agg.items()):
        print(f"  {m}: calls={a[0]} completion={a[1]} reasoning={a[2]} share={a[2] / a[1] * 100:.1f}% length={a[3]} (of which status bad_response={a[4]})")
    print("  calls with reasoning_tokens == 0:", sum(1 for r in f if r["reasoning_tokens"] == 0))

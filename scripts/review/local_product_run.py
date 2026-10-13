"""Run one catalog entry through the same product configuration as a live quick check, from this machine.

Mirrors `api.make_orchestrator(quick=True)` + `catalog.run_candidate`; writes every event to
<out>/events.jsonl and keeps the local LLM cost log (finish_reason, token counts) for diagnosis.
Paid: Token Factory per .env. Does not write benchmark/catalog.yaml.

Usage: uv run python scripts/review/local_product_run.py <catalog_id> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent import api, catalog as cat, live_catalog  # noqa: E402

QUIET = {"llm_call", "code_text", "plan_text", "tavily_call"}


def main() -> int:
    catalog_id, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    entry = next((e for e in cat.load_catalog(api.CATALOG_PATH) if e["id"] == catalog_id), None) \
        or live_catalog.find_entry(catalog_id)
    if entry is None:
        print("unknown catalog id", catalog_id)
        return 1
    eligible = live_catalog.eligibility(entry)
    if not eligible["runnable"]:
        print("not runnable:", eligible["reason"])
        return 1
    orch = api.make_orchestrator(True, True)
    log = (out / "events.jsonl").open("a")
    t0 = time.time()

    def on_event(evt):
        log.write(json.dumps(evt, default=str) + "\n")
        log.flush()
        if evt.get("type") not in QUIET:
            print(f"{time.time() - t0:7.1f} {evt.get('type')} {json.dumps(evt, default=str)[:240]}", flush=True)

    orch.on_event = on_event
    outcome = cat.run_candidate(orch, entry)
    summary = {"status": outcome.status, "reason": getattr(outcome, "reason", None),
               "run_dir": str(getattr(outcome, "run_dir", "")), "seconds": round(time.time() - t0, 1),
               "spent_net_usd": round(getattr(orch, "spent_usd", 0.0), 5)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

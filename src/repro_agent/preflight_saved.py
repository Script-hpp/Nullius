"""Readiness check for saved discovery claims that have none yet, e.g. claims found before 2026-10-15.

Since 2026-10-15 a found claim gets a start button only after its readiness check passed (target bound to its
table, authors' code read, spec without conflict). New searches check their claims themselves; this script checks
the claims saved earlier. Run it where the server's logs/ volume is mounted (inside the container):

    python -m repro_agent.preflight_saved                 # list the claims without a check, no calls
    python -m repro_agent.preflight_saved --run           # check all of them (paid: model, Tavily, sandbox)
    python -m repro_agent.preflight_saved --run <id> ...  # check only these claims

Each check has its own spending limit (MAX_COST_PER_PREFLIGHT_USD, default 0.35 $ gross).
"""

import sys

from repro_agent import api, live_catalog


def main(argv: list[str]) -> int:
    run = "--run" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    entries = live_catalog.load_entries()
    todo = [e for e in entries if (e["id"] in wanted if wanted else not e.get("readiness"))
            and live_catalog.eligibility(e)["runnable"]]
    if not todo:
        print("No saved claim needs a readiness check.")
        return 0
    for entry in todo:
        print(f"{entry['id']}  {entry.get('title', '')[:70]}")
        if not run:
            continue
        readiness = api.preflight_claim(entry["id"])
        state = "ready" if readiness["ready"] else f"not offered ({readiness['stage']}): {readiness['reason']}"
        print(f"  -> {state}  [{readiness.get('cost_gross_usd')} $]", flush=True)
    if not run:
        print(f"\n{len(todo)} claim(s) without a readiness check. Add --run to check them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

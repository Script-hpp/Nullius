"""Validiert die Graph-Ablation an den 10 GCN-Fixture-Codes (kein LLM). Regel: benchmark/experiments/ablation_gcn.md.

  uv run python scripts/validate_ablation.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import get_sandbox
from repro_agent.tasks import cora_task

FIX = REPO_ROOT / "tests" / "fixtures"
USED, UNUSED = "Graph genutzt", "Graph nicht genutzt"
CASES = {
    "agent_codes/gcn_cached_faithful": USED, "agent_codes/gcn_sparse_selfloop": USED,
    "agent_codes/gcn_edge_selfloop": USED, "agent_codes/gcn_variant_no_selfloops": USED,
    "agent_codes/gcn_variant_adjacency_unused": UNUSED,
    "agent_codes_v2/gcn_new_faithful_dense": USED, "agent_codes_v2/gcn_new_faithful_scatter": USED,
    "agent_codes_v2/gcn_new_row_normalised": USED, "agent_codes_v2/gcn_new_one_layer": USED,
    "agent_codes_v2/gcn_new_adjacency_stored_unused": UNUSED,
}


def main() -> None:
    orch = Orchestrator.__new__(Orchestrator)
    orch.sandbox, orch._smoke_ok = get_sandbox("nebius"), set()
    task = cora_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds)

    def run(item):
        name, expected = item
        code = (FIX / f"{name}.py").read_text()
        full = orch._run_and_eval(orch.sandbox.branch(snap), code, 0, task, "accuracy", ablate=False)
        # Vollversion ohne Smoke-Pruefung: Fixture-Codes kennen SMOKE nicht
        return name, expected, full, orch

    # Vollversion: Smoke-Pruefung ueberspringen, indem der Code als bestanden gilt
    import hashlib
    for name in CASES:
        orch._smoke_ok.add(hashlib.sha256((FIX / f"{name}.py").read_text().encode()).hexdigest())
    with ThreadPoolExecutor(5) as pool:
        fulls = list(pool.map(run, CASES.items()))

    def ablate(item):
        name, expected, full, _ = item
        if full["status"] != "ok":
            return name, expected, full.get("measured"), None, f"Vollversion {full['status']}: {str(full.get('detail'))[-200:]}"
        ab = orch.ablation_check(snap, (FIX / f"{name}.py").read_text(), 0, task, "accuracy", full["measured"])
        return name, expected, full["measured"], ab, ""

    with ThreadPoolExecutor(5) as pool:
        rows = list(pool.map(ablate, fulls))
    hits = 0
    for name, expected, acc, ab, err in rows:
        got = ab["verdict"] if ab else "Vollversion fehlgeschlagen"
        ok = got == expected
        hits += ok
        abl = "n/a" if not ab or ab["accuracy_ablated"] is None else f"{ab['accuracy_ablated']:.4f}"
        print(f"{'OK ' if ok else 'ABW'} {name:48s} mit Graph {acc if acc is None else round(acc, 4)} | ohne Kanten {abl} | "
              f"{got} (erwartet {expected}) {err or (ab or {}).get('detail') or ''}")
    print(f"\n{hits}/{len(rows)} wie erwartet eingeordnet. Regel: alle 10, kein unklar/nicht messbar.")


if __name__ == "__main__":
    main()

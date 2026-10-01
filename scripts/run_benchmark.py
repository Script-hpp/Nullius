"""Benchmark-Lauf. v1: benchmark/experiments/benchmark_protocol.md (Standard, unveraendert). v2: benchmark/experiments/benchmark_v2.md (`--v2`).

Pro Paper und Arm 3 frische Generierungen; je Generierung der echte Claim und der Placebo-Claim. Arme abwechselnd, Ergebnisse als JSONL,
Neustart macht weiter.
  LLM_PROVIDER=tokenfactory uv run python scripts/run_benchmark.py <paper_id> [<paper_id> ...]
  ... run_benchmark.py --v2 [--dry-run] [<paper_id> ...]   (v2: Produktkonfiguration graph_repair=True, Store runs/benchmark_v2.jsonl)
  ... run_benchmark.py [--v2] --dry-run                     (Laufliste, Seeds, Fixture-Abdeckung, geschaetzte Kosten und Dauer; keine Aufrufe)
Arm A: ohne Tavily (`use_tavily=False`, keine Rollen 2 und 4). Arm B: mit Tavily (Rolle 2 Zielwert-Beleg, Rolle 4 Setup-Details).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from repro_agent.claims_source import FIXTURE_DIR
from repro_agent.dossier import REPO_ROOT
from repro_agent.llm import BillingBlocked
from repro_agent.loop import Orchestrator
from repro_agent.repo_source import _repo_key, find_repo_url

CODE_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
DIRTY = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
N_GEN = 3
STORE = REPO_ROOT / "runs" / "benchmark.jsonl"
STORE_V2 = REPO_ROOT / "runs" / "benchmark_v2.jsonl"
STORE_V3 = REPO_ROOT / "runs" / "benchmark_v3.jsonl"   # v3: wie v2, Code nach Fehleranalyse (benchmark_v3.md)
V2_BUDGET_GROSS_USD = 6.0    # benchmark_v2.md: kein neuer Lauf, wenn die Bruttokosten im v2-Store diese Grenze erreicht haben
V2_VAT = 0.19
# Annahmen der Schaetzung (Dry-Run); v2 mit Reparaturschleifen teurer als v1 -> Faktor, Dauer nicht gemessen
COST_FALLBACK_NET = {"A_ohne_tavily": 0.019, "B_mit_tavily": 0.053}   # Mittel je Lauf aus v1 (results/benchmark_results.md)
V2_COST_FACTOR = 1.5
MINUTES_PER_RUN = 25         # Schaetzung (benchmark_plan.md, Abschnitt 5), nicht gemessen
CLAIMS = {   # echter Claim, Placebo-Claim
    "kipf_gcn_2017": ("gcn_cora_acc", "gcn_cora_acc_placebo"),
    "shchur_pitfalls_2018": ("shchur_gcn_cora_acc", "shchur_gcn_cora_acc_placebo"),
    "zeng_dlinear_2023": ("dlinear_etth1_96_mse", "dlinear_etth1_96_mse_placebo"),
    "wu_sgc_2019": ("sgc_cora_acc", "sgc_cora_acc_placebo"),
}
CLAIMS_V2 = {   # alle v1-Papers laufen neu, dazu GAT, DGI, Shchur-GAT, NLinear (benchmark_v2.md)
    **CLAIMS,
    "velickovic_gat_2018": ("gat_cora_acc", "gat_cora_acc_placebo"),
    "velickovic_dgi_2019": ("dgi_cora_acc", "dgi_cora_acc_placebo"),
    "shchur_gat_2018": ("shchur_gat_cora_acc", "shchur_gat_cora_acc_placebo"),
    "zeng_nlinear_2023": ("nlinear_etth1_96_mse", "nlinear_etth1_96_mse_placebo"),
}
ARMS = ("A_ohne_tavily", "B_mit_tavily")


def fixture_status(paper: dict, fixture_dir: Path = FIXTURE_DIR) -> list[str]:
    """Fixture-Luecken, die im Modus fixture einen Live-Tavily-Aufruf ausloesen wuerden (nur Arm B). Keine Aufrufe."""
    arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
    text_file = fixture_dir / f"{arxiv_id}.json"
    if not text_file.exists():
        return [f"Paper-Text {arxiv_id}.json fehlt (Live-Extract in Rolle 2)"]
    url = find_repo_url(json.loads(text_file.read_text())["text"])
    if url and not (fixture_dir / "repos" / f"{_repo_key(url)}_readme.json").exists():
        return [f"README {_repo_key(url)}_readme.json fehlt (Live-Extract in Rolle 4, Arm B)"]
    return []


def mean_cost_net(rows: list[dict]) -> dict[str, float]:
    """Mittel der Nettokosten je Lauf und Arm aus bisherigen Zeilen (ein Wert je Lauf), sonst Annahmen."""
    per_run = {(r["paper"], r["arm"], r["generation"]): r["cost_usd_run"] for r in rows if "cost_usd_run" in r}
    out = {}
    for arm in ARMS:
        vals = [c for (_, a, _), c in per_run.items() if a == arm]
        out[arm] = statistics.mean(vals) if vals else COST_FALLBACK_NET[arm]
    return out


def pending_runs(papers: list[str], rows: list[dict], n_gen: int = N_GEN) -> list[tuple[str, str, int]]:
    """Laeufe in der Reihenfolge des Skripts (je Paper: Generierung, dann Arme abwechselnd), ohne bereits vorhandene Zeilen."""
    return [(p, arm, g) for p in papers for g in range(n_gen) for arm in ARMS
            if not any(r["paper"] == p and r["arm"] == arm and r["generation"] == g for r in rows)]


def gross_spent(rows: list[dict], vat: float = V2_VAT) -> float:
    """Bruttokosten der bisherigen Laeufe im Store (cost_usd_run steht in jeder Zeile eines Laufs, je Lauf einmal zaehlen)."""
    per_run = {r["run_dir"]: r["cost_usd_run"] for r in rows if "run_dir" in r and "cost_usd_run" in r}
    return sum(per_run.values()) * (1 + vat)


def dry_run(v2: bool, papers: list[str], rows: list[dict], paper_specs: dict[str, dict], claims: dict) -> str:
    """Bericht ueber geplante Laeufe. Keine LLM-, Tavily- oder Sandbox-Aufrufe."""
    runs = pending_runs(papers, rows)
    v1_rows = [json.loads(x) for x in STORE.read_text().splitlines()] if STORE.exists() else []
    mean = mean_cost_net(rows or v1_rows)   # v2-Schaetzung vor dem ersten Lauf aus v1
    factor = V2_COST_FACTOR if v2 else 1.0
    lines = [f"Benchmark {'v2 (graph_repair=True)' if v2 else 'v1 (graph_repair=False)'}; Tavily-Modus fixture; Store {(STORE_V2 if v2 else STORE).name}",
             f"{len(runs)} offene Laeufe ({len(papers)} Papers x {len(ARMS)} Arme x {N_GEN} Generierungen, bereits vorhandene abgezogen)"]
    for p in papers:
        spec = paper_specs[p]
        todo = [(a, g) for (pp, a, g) in runs if pp == p]
        miss = fixture_status(spec)
        lines.append(f"  {p}: Claims {list(claims[p])}, Seeds {spec.get('seeds', 5)}, Task {spec['dataset']}, offen {len(todo)}"
                     + (f"; ACHTUNG Fixture-Luecke: {'; '.join(miss)}" if miss else "; Fixtures vollstaendig"))
    net = sum(mean[a] * factor for _, a, _ in runs)
    lines.append(f"Geschaetzte Kosten: {net:.2f} $ netto, {net * (1 + V2_VAT):.2f} $ brutto (Mittel je Lauf A {mean['A_ohne_tavily']:.3f} $, "
                 f"B {mean['B_mit_tavily']:.3f} $ netto" + (f", Faktor {factor} fuer Reparaturschleifen" if v2 else "") + ")")
    hours = len(runs) * MINUTES_PER_RUN / 60
    lines.append(f"Geschaetzte Dauer: {hours:.0f} h seriell, etwa {hours / 3:.0f} h mit 3 parallelen Prozessen (Annahme {MINUTES_PER_RUN} min je Lauf, nicht gemessen)")
    if v2:
        lines.append(f"Budget-Grenze v2: {V2_BUDGET_GROSS_USD:.2f} $ brutto; bisher im Store {gross_spent(rows):.2f} $")
    lines.append("Keine Aufrufe ausgefuehrt.")
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    v3 = "--v3" in sys.argv
    v2, dry = "--v2" in sys.argv or v3, "--dry-run" in sys.argv
    claims_map, store = (CLAIMS_V2, STORE_V3 if v3 else STORE_V2) if v2 else (CLAIMS, STORE)
    papers = args or list(claims_map)
    all_papers = yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
    yaml_targets = {c["id"]: c["target"] for p in all_papers for c in p.get("claims", [])}
    rows = [json.loads(x) for x in store.read_text().splitlines()] if store.exists() else []
    if dry:
        print(dry_run(v2, papers, rows, {p["id"]: p for p in all_papers}, claims_map))
        return
    for paper in papers:
        claim_ids = list(claims_map[paper])
        kw = {"graph_repair": True} if v2 else {}
        arms = {
            "A_ohne_tavily": Orchestrator(use_tavily=False, fresh=True, setup_details=False, claim_ids=claim_ids, tavily_mode="fixture", **kw),
            "B_mit_tavily": Orchestrator(use_tavily=True, fresh=True, setup_details=True, claim_ids=claim_ids, tavily_mode="fixture", **kw),
        }
        snapshots: dict = {}
        for orch in arms.values():
            orch._snapshots = snapshots
        for g in range(N_GEN):
            for arm, orch in arms.items():
                if any(r["paper"] == paper and r["arm"] == arm and r["generation"] == g for r in rows):
                    continue
                if v2 and gross_spent(rows) >= V2_BUDGET_GROSS_USD:   # Abbruchkriterium benchmark_v2.md
                    print(f"Budget-Grenze v2 erreicht ({gross_spent(rows):.2f} $ brutto >= {V2_BUDGET_GROSS_USD:.2f} $). Stopp vor {paper} {arm} Generierung {g}.")
                    return
                base = {"paper": paper, "arm": arm, "generation": g, "code_commit": CODE_COMMIT + ("+dirty" if DIRTY else ""), **({"config": "v3_graph_repair" if v3 else "v2_graph_repair"} if v2 else {})}
                try:
                    d = orch.run(paper)
                except BillingBlocked:
                    raise
                except RuntimeError as exc:   # z. B. Kostenlimit, ungueltige Extraktion: als Ausfall festhalten, weiter
                    new = [{**base, "claim": cid, "origin": "placebo" if cid.endswith("_placebo") else "paper",
                            "urteil": "abgebrochen", "error": str(exc)[:300]} for cid in claim_ids]
                else:
                    new = []
                    for v in d.verdicts:
                        claim = next(c for c in d.claims if c.id == v.claim_id)
                        new.append({
                            **base, "claim": v.claim_id, "origin": claim.origin, "urteil": v.urteil, "mean": v.measured_mean,
                            "std": v.measured_std, "target": claim.target, "target_matches_yaml": abs(claim.target - yaml_targets[v.claim_id]) < 1e-9,
                            "gesamt": v.gesamtaussage, "methode": v.methode,
                            "ablation": d.ablations.get(v.claim_id, {}).get("verdict"), "cost_usd_run": round(orch.spent_usd, 4),
                            "run_dir": str(orch.last_run_dir.relative_to(REPO_ROOT)),
                        })
                for r in new:
                    rows.append(r)
                    with store.open("a") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print("ERGEBNIS", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

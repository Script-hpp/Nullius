"""Modus Nachlauf, Meilenstein M3: Nemotron liest das Repo, waehlt Startbefehl und Seed-Weg, schreibt den Adapter; der Harness misst.

Der Reviewer startet es (echte Token-Factory- und Sandbox-Aufrufe). Die Entwicklung laeuft nur mit Fake-LLM und Fake-Sandbox
(tests/test_nachlauf_agent.py). Kein Tavily: Repo-Adresse und Commit werden vorgegeben.
  uv run python scripts/run_nachlauf.py --paper rlinear_2023 --repo https://github.com/<owner>/<repo> --commit <hash> --dry-run
  uv run python scripts/run_nachlauf.py --paper rlinear_2023 --repo ... --commit ... [--seeds repo,0,1,2,3] [--offsets 8]
                                          [--store runs/nachlauf_dev.jsonl] [--budget-stop 3.0]
Claim, Zielwert und Toleranz stehen in benchmark/papers.yaml. Je Lauf eine Zeile im Store (Standard runs/nachlauf_v1.jsonl); das Kostenlimit
(--budget-stop, brutto, Standard 3 $) wird vor jedem LLM-Aufruf gegen die Kosten dieses Prozesses UND den Store geprueft.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent import nachlauf_agent as na  # noqa: E402
from repro_agent.harness import independence  # noqa: E402

DEFAULT_STORE = ROOT / "runs" / "nachlauf_v1.jsonl"
DEFAULT_BUDGET_STOP = 3.0   # brutto, Vorab-Festlegung nachlauf_v1.md
VAT = 0.19


def code_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    return head + ("+dirty" if dirty else "")


def load_claim(paper_id: str, claim_id: str | None):
    from repro_agent.loop import load_claims
    paper, claims = load_claims(paper_id)
    if paper.get("dataset") != "ETTh1":
        raise SystemExit(f"Nachlauf v1 gilt nur fuer ETTh1 (Paper {paper_id}: {paper.get('dataset')})")
    cands = [c for c in claims if c.origin == "paper" and (claim_id is None or c.id == claim_id)]
    if len(cands) != 1:
        raise SystemExit(f"genau ein Paper-Claim erwartet, gefunden: {[c.id for c in cands]} (--claim angeben)")
    return paper, cands[0]


def default_factory():
    """Echte Sandbox-Fabrik, derselbe Importpfad wie im Betrieb (tests/test_nachlauf_m2.py und test_nachlauf_agent.py importieren ihn ohne Client)."""
    from repro_agent.sandbox.base import get_sandbox
    return lambda: get_sandbox("nebius")


def result_row(paper_id: str, claim, repo: str, commit: str, dossier, agent: "na.NachlaufAgent", error: str | None = None) -> dict:
    net = round(agent.guard.run_net(), 5)
    base = {"paper": paper_id, "claim": claim.id, "unit": repo.rstrip("/").split("/")[-1].removesuffix(".git"), "repo": repo, "commit": commit,
            "code_commit": code_commit(), "cost_usd_run": net, "cost_usd_run_gross": round(net * (1 + VAT), 5), "n_llm_calls": len(agent.asker.calls)}
    if dossier is None:
        return {**base, "urteil": "abgebrochen", "valid": False, "error": (error or "")[:300]}
    v = dossier.verdicts[0]
    info = next(iter(dossier.nachlauf.values()))
    rejected = sorted({c for v_ in info.get("rejected_by", {}).values() for c in v_})
    return {**base, "urteil": v.urteil, "mean": v.measured_mean, "std": v.measured_std, "target": claim.target,
            "valid": v.urteil != "nicht reproduzierbar (Setup)", "rejected_by": rejected, "outcome": (info.get("agent") or {}).get("outcome"),
            "warnings": dossier.warnings, "command": info.get("command"), "command_documented": info.get("command_documented")}


def main(argv: list[str] | None = None, sandbox_factory=None, llm_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True, help="Paper-ID aus benchmark/papers.yaml (ETTh1)")
    ap.add_argument("--claim", help="Claim-ID, falls das Paper mehrere Paper-Claims hat")
    ap.add_argument("--repo", required=True, help="Repo-URL (vorgegeben, kein Tavily)")
    ap.add_argument("--commit", required=True, help="fester Commit-Hash")
    ap.add_argument("--seeds", default="repo,0,1,2,3", help="Komma-Liste, `repo` = Seed des Repos")
    ap.add_argument("--offsets", type=int, default=independence.N_OFFSETS, help="P2-Versaetze je Seed")
    ap.add_argument("--store", default=str(DEFAULT_STORE), help="Store (jsonl), eine Zeile je Lauf")
    ap.add_argument("--budget-stop", type=float, default=DEFAULT_BUDGET_STOP, help="Kostenstopp brutto in $ (Prozess + Store)")
    ap.add_argument("--dry-run", action="store_true", help="nichts aufrufen (kein LLM, keine Sandbox), nur den Plan drucken")
    a = ap.parse_args(argv)
    paper, claim = load_claim(a.paper, a.claim)
    store = Path(a.store)
    seeds = [t.strip() if t.strip() == "repo" else int(t) for t in a.seeds.split(",") if t.strip()]
    spent = na.store_gross(store, VAT)
    if a.dry_run:
        print(f"Paper {a.paper} ({paper['title']}), Claim {claim.id}: {claim.text} | Ziel {claim.target} +/- {claim.tolerance} ({claim.metric})")
        print(f"Repo {a.repo} @ {a.commit}; Seeds {seeds}; P2: {a.offsets} Versaetze je Seed (parallel {na.nl.PARALLEL}); Erkundung mit {na.EXPLORE_OFFSETS}")
        print(f"Store {store}: bisher {spent:.4f} $ brutto; Stopp bei {a.budget_stop:.2f} $ brutto -> {'WUERDE SOFORT STOPPEN' if spent >= a.budget_stop else 'ok'}")
        print("Schritte: 1 Lese-Snapshot (Klon, 1 LLM-Aufruf: Dateiauswahl) | 2 Startbefehl (1 Aufruf, bei unbelegten Zitaten +1) | "
              "3 Erkundung: Training (Reparatur: je 1 Aufruf, max 3) und Adapter (1 + Reparaturen, max 3) | 4 Seeds (Training, Adapter, P3, P2, Harness)")
        print("LLM-Aufrufe je Lauf: Normalfall 3 bis 4, im ungünstigsten Fall bis etwa 11 (Rollen: code, fuer die Dateiauswahl extract = Modell von code; kein Ultra)")
        return 0
    if sandbox_factory is None:
        sandbox_factory = default_factory()
    if llm_factory is None:
        def llm_factory():
            from repro_agent.config import load_settings
            from repro_agent.llm import LLMClient
            return LLMClient(dataclasses.replace(load_settings(), llm_cache=False))   # Nachlauf-Laeufe: LLM-Cache aus
    llm = llm_factory()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / "nachlauf" / a.paper / ts
    agent = na.NachlaufAgent(sandbox_factory(), llm, store=store, budget_stop=a.budget_stop, out_dir=out)
    dossier, error = None, None
    try:
        dossier = agent.run(a.paper, claim, a.repo, a.commit, seeds=seeds, offsets=a.offsets)
    except (na.BudgetStop, na.AgentAnswerError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    row = result_row(a.paper, claim, a.repo, a.commit, dossier, agent, error)
    if dossier is not None:
        (out / "dossier.json").write_text(json.dumps(dataclasses.asdict(dossier), indent=2, ensure_ascii=False, default=str))
        row["run_dir"] = str(out.relative_to(ROOT))
    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    agent.guard.note_stored(row["cost_usd_run"])
    print("ERGEBNIS", json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if dossier is not None and row["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

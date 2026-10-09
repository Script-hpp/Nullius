"""recall cg v1: Laufskript. Vorab-Festlegung: benchmark/experiments/recall_cg_v1.md (committet VOR dem ersten Modellaufruf).

  uv run python scripts/run_recall_cg_v1.py --dry-run              Laufliste, Fixture-Abdeckung, Kosten, Dauer (keine Aufrufe)
  uv run python scripts/run_recall_cg_v1.py --mechanics            Schleife auf Nebius mit skriptgesteuertem "Modell" (kein Modellaufruf, kein Tavily)
  LLM_PROVIDER=tokenfactory uv run python scripts/run_recall_cg_v1.py --prep   einmalig: Fixtures (S4-Repo-Auszug je Einheit, RLinear) in die Overlay-Kopie
  nohup env LLM_PROVIDER=tokenfactory uv run python scripts/run_recall_cg_v1.py --run &       Stufe 1, Gate, Stufe 2 (Neustart setzt fort)
  uv run python scripts/run_recall_cg_v1.py --worker <Einheit> <Generation> [--placebo]       ein Lauf (vom Planer gestartet)

Fixtures: Overlay `runs/recall_cg_v1_fixtures/` = Kopie von benchmark/fixtures plus benchmark/experiments/recall_cg_v1_fixtures/. In benchmark/fixtures
wird nie geschrieben. Kein Tavily-Aufruf (die Zugriffsfunktion des Orchestrators wirft); ein Worker startet nur, wenn alle Fixtures da sind.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import recall_cg_v1 as R  # noqa: E402

OVERLAY = ROOT / "runs" / "recall_cg_v1_fixtures"
EXTRAS = ROOT / "benchmark" / "experiments" / "recall_cg_v1_fixtures"
STATE = ROOT / "runs" / "recall_cg_v1_state.json"
LOGDIR = ROOT / "runs" / "recall_cg_v1_logs"
INCIDENTS = ROOT / "runs" / "recall_cg_v1_incidents.jsonl"
STOPFILE = ROOT / "runs" / "recall_cg_v1.STOP"
GATEFILE = ROOT / "runs" / "recall_cg_v1_gate.json"
RLINEAR_PAPER = Path("/home/onuralp/Desktop/hackathon/research/runs/rlinear_2023/20261002-105416/paper_text.json")   # gespeicherter Lauf vom 2026-10-02
MAX_INFRA_RETRIES = 3


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- Overlay und Fixture-Abdeckung ------------------------------------------------------------------------------------------------
def build_overlay() -> Path:
    """Overlay = benchmark/fixtures + committete Zusaetze. Vorhandene Dateien im Overlay bleiben (Prep schreibt dorthin)."""
    shutil.copytree(ROOT / "benchmark" / "fixtures", OVERLAY, dirs_exist_ok=True)
    if EXTRAS.exists():
        shutil.copytree(EXTRAS, OVERLAY, dirs_exist_ok=True)
    return OVERLAY


def patch_fixture_dirs() -> None:
    """Alle Quellmodule lesen und schreiben im Overlay (Module-Variablen werden beim Aufruf gelesen)."""
    from repro_agent import claims_source, repo_source
    claims_source.FIXTURE_DIR = OVERLAY
    repo_source.FIXTURE_DIR = OVERLAY / "repos"


def _paper(unit: str) -> dict:
    from repro_agent.loop import load_claims
    return load_claims(unit)[0]


def fixtures_missing(unit: str, ids: list[str]) -> list[str]:
    """Fixtures, deren Fehlen im Modus fixture einen Tavily-Aufruf, einen Klon oder einen Modellaufruf der Vorbereitung ausloesen wuerde. Keine Aufrufe."""
    from repro_agent.loop import claim_specs
    from repro_agent.repo_source import _repo_key, find_repo_url, unit_fixture_name
    paper = _paper(unit)
    arxiv_id = paper["arxiv_url"].rstrip("/").split("/abs/")[-1]
    text_file = OVERLAY / f"{arxiv_id}.json"
    if not text_file.exists():
        return [f"Paper-Text {arxiv_id}.json"]
    pt = json.loads(text_file.read_text())
    missing = []
    url = find_repo_url(pt["text"])
    if not url:   # Rolle 1 (Produktpfad): Repo-Suche aus der Fixture search_<arxiv_id>.json (sonst ein Tavily-Aufruf)
        search = OVERLAY / "repos" / f"search_{arxiv_id}.json"
        if search.exists():
            url = json.loads(search.read_text()).get("url")
        else:
            missing.append(f"Repo-Suche search_{arxiv_id}.json (Rolle 1)")
    if url:
        if not (OVERLAY / "repos" / f"{_repo_key(url)}_readme.json").exists():
            missing.append(f"README {_repo_key(url)}_readme.json")
        if not (OVERLAY / "repos" / unit_fixture_name(url, unit)).exists():
            missing.append(f"S4-Repo-Auszug {unit_fixture_name(url, unit)}")
    for cid, spec in ((cid, claim_specs(paper).get(cid, {}).get("extract")) for cid in ids):
        if spec:
            raw = f"{pt['sha256']}|{spec['method']}|{spec['dataset']}|{spec['metric']}" + ("" if spec.get("percent", True) else "|raw")
            key = hashlib.sha256(raw.encode()).hexdigest()[:16]
            if not (OVERLAY / f"{arxiv_id}_target_{key}.json").exists():
                missing.append(f"Zielwert {arxiv_id}_target_{key}.json ({cid})")
    return missing


# ---- Ein Lauf --------------------------------------------------------------------------------------------------------------------
def _is_infra(exc: BaseException) -> bool:
    import httpx
    from repro_agent.sandbox.base import SandboxSetupError
    if str(exc).startswith("LLM-Aufruf fehlgeschlagen"):   # Nachtrag 1: API-Ausfall nach 5 Versuchen (z. B. 404 durch entferntes Modell) ist Infrastruktur, kein Agenten-Fehler
        return True
    return isinstance(exc, (SandboxSetupError, httpx.TransportError)) or type(exc).__name__ in ("ApiTimeoutError", "ConnectionError", "TimeoutError")


def _forbidden_tavily():
    raise RuntimeError("recall cg v1: Tavily-Aufruf verboten (Fixtures)")


def append_rows(rows: list[dict]) -> None:
    R.STORE.parent.mkdir(parents=True, exist_ok=True)
    with R.STORE.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def worker(unit: str, gen: int, placebo: bool) -> int:
    import yaml
    from repro_agent.costs import BudgetExceeded
    from repro_agent.dossier import REPO_ROOT
    from repro_agent.llm import BillingBlocked
    from repro_agent.loop import Orchestrator

    job = (unit, gen, placebo)
    ids = R.claim_ids(job)
    build_overlay()
    patch_fixture_dirs()
    miss = fixtures_missing(unit, ids)
    if miss:
        print("FIXTURE-LUECKE:", "; ".join(miss), flush=True)
        return R.EX_FIXTURE
    papers = yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
    yaml_targets = {c["id"]: c["target"] for p in papers for c in p.get("claims", [])}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    orch = Orchestrator(claim_ids=ids, **R.ORCH_KWARGS)
    s = orch.settings
    assert s.llm.provider == "tokenfactory" and s.sandbox_backend == "nebius" and not s.llm_cache, "recall cg v1 laeuft nur auf Token Factory, Nebius, ohne Cache"
    orch._tavily_getter = lambda: _forbidden_tavily
    orch.train_timeout_s = R.TRAIN_TIMEOUTS.get(unit, 600)
    base = {"paper": unit, "arm": "B_mit_tavily", "generation": gen, "config": R.CONFIG, "mode": R.MODE,
            "code_commit": commit + ("+dirty" if dirty else ""), "started": now_iso()}
    try:
        d = orch.run(unit)
    except BillingBlocked as exc:
        print("BILLING:", exc, flush=True)
        return R.EX_BILLING
    except BudgetExceeded as exc:
        print("BUDGET-STOPP:", exc, flush=True)
        return R.EX_BUDGET
    except RuntimeError as exc:
        if _is_infra(exc):
            print("INFRASTRUKTUR:", type(exc).__name__, str(exc)[:300], flush=True)
            return R.EX_INFRA
        rows = [{**base, "claim": cid, "origin": "placebo" if cid.endswith("_placebo") else "paper", "urteil": "abgebrochen", "error": str(exc)[:300],
                 "cost_usd_run": round(orch.spent_usd, 4)} for cid in ids]
        append_rows(rows)
        print("ABGEBROCHEN:", exc, flush=True)
        return R.EX_OK
    except Exception as exc:   # noqa: BLE001
        if _is_infra(exc):
            print("INFRASTRUKTUR:", type(exc).__name__, str(exc)[:300], flush=True)
            return R.EX_INFRA
        raise
    rows = []
    for v in d.verdicts:
        claim = next(c for c in d.claims if c.id == v.claim_id)
        rows.append({**base, "claim": v.claim_id, "origin": claim.origin, "urteil": v.urteil, "mean": v.measured_mean, "std": v.measured_std,
                     "target": claim.target, "target_matches_yaml": abs(claim.target - yaml_targets[v.claim_id]) < 1e-9, "gesamt": v.gesamtaussage,
                     "methode": v.methode, "ablation": d.ablations.get(v.claim_id, {}).get("verdict"), "cost_usd_run": round(orch.spent_usd, 4),
                     "run_dir": str(orch.last_run_dir.relative_to(REPO_ROOT)), "finished": now_iso()})
    append_rows(rows)
    for r in rows:
        print("ERGEBNIS", json.dumps(r, ensure_ascii=False), flush=True)
    return R.EX_OK


# ---- Planer ------------------------------------------------------------------------------------------------------------------------
def spent_since(since_iso: str) -> float:
    from repro_agent import llm
    from repro_agent.costs import logged_gross_total
    return logged_gross_total(llm.COST_LOG_PATH, R.VAT, since_iso)


def start_time() -> str:
    if STATE.exists():
        return json.loads(STATE.read_text())["start"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"start": now_iso(), "stop_gross": R.STOP_GROSS_USD, "cap_gross": R.CAP_GROSS_USD}))
    return json.loads(STATE.read_text())["start"]


def log_incident(job: R.Job, rc: int, attempt: int) -> None:
    with INCIDENTS.open("a") as f:
        f.write(json.dumps({"ts": now_iso(), "job": list(job), "returncode": rc, "attempt": attempt, "code_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}) + "\n")


def run_queue(jobs: list[R.Job], since: str) -> str:
    """Startet Jobs (hoechstens WORKERS gleichzeitig) mit Reservierungspruefung. Rueckgabe: ok | budget | billing | stop."""
    pending = [j for j in jobs if not R.done(R.load_rows(), j)]
    running: dict[R.Job, subprocess.Popen] = {}
    attempts: dict[R.Job, int] = {}
    outcome = "ok"
    env = {**os.environ, **R.worker_env(since)}
    LOGDIR.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for job, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[job]
            if rc == R.EX_OK:
                print(f"[{now_iso()}] fertig {job}", flush=True)
            elif rc == R.EX_INFRA and attempts.get(job, 0) < MAX_INFRA_RETRIES:
                log_incident(job, rc, attempts.get(job, 0))
                print(f"[{now_iso()}] Infrastruktur-Abbruch {job}, Wiederholung", flush=True)
                pending.insert(0, job)
            elif rc == R.EX_BILLING:
                outcome = "billing"
                print(f"[{now_iso()}] BillingBlocked bei {job}: Stopp", flush=True)
            elif rc == R.EX_BUDGET:
                outcome = outcome if outcome == "billing" else "budget"
                print(f"[{now_iso()}] Budget-Stopp im Aufruf bei {job}", flush=True)
            else:
                log_incident(job, rc, attempts.get(job, 0))
                print(f"[{now_iso()}] Job {job} endete mit Code {rc} (kein Ergebnis, siehe {LOGDIR})", flush=True)
        halted = outcome in ("billing", "budget") or STOPFILE.exists()
        if STOPFILE.exists() and outcome == "ok":
            outcome = "stop"
        while pending and not halted and len(running) < R.WORKERS:
            spent = spent_since(since)
            nxt = pending[0]
            if not R.can_start(spent, list(running), nxt):
                if not running:
                    print(f"[{now_iso()}] Reservierung: {spent:.2f} $ + Reserve > {R.STOP_GROSS_USD:.2f} $, nichts laeuft: Budget-Stopp", flush=True)
                    return "budget"
                break
            pending.pop(0)
            attempts[nxt] = attempts.get(nxt, 0) + 1
            cmd = [sys.executable, str(Path(__file__)), "--worker", nxt[0], str(nxt[1])] + (["--placebo"] if nxt[2] else [])
            log = (LOGDIR / f"{nxt[0]}_g{nxt[1]}_a{attempts[nxt]}.log").open("w")
            running[nxt] = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"[{now_iso()}] Start {nxt} (Versuch {attempts[nxt]}), laufend {len(running)}, Verbrauch {spent:.2f} $ brutto", flush=True)
        if halted and not running:
            return outcome
        if not pending and not running:
            break
        time.sleep(20)
    return outcome


def run_all() -> int:
    refs = R.reference_verdicts()
    since = start_time()
    print(f"[{now_iso()}] recall cg v1, Beginn der Kostenzaehlung {since}, Stopp {R.STOP_GROSS_USD} $ brutto (Deckel {R.CAP_GROSS_USD} $)", flush=True)
    build_overlay()
    patch_fixture_dirs()
    for unit in R.ALL_UNITS:
        miss = fixtures_missing(unit, R.claim_ids((unit, 0, True)))
        if miss:
            print(f"Fixture-Luecke {unit}: {miss}. Erst --prep.", flush=True)
            return 1
    out = run_queue(R.stage1_jobs(), since)
    rows = R.load_rows()
    g = R.gate(rows, refs)
    if out == "budget":   # Nachtrag 1: Stopp erreicht, bevor alle vorgesehenen Laeufe fertig sind => PAUSIERT, nicht finalisiert
        print(f"[{now_iso()}] PAUSIERT: Budget-Stopp in Stufe 1 ({spent_since(since):.2f} $ brutto). Keine Auswertung, Meldung an die Koordination.", flush=True)
        return 0
    GATEFILE.write_text(json.dumps({**g, "ts": now_iso(), "spent_gross": spent_since(since), "stage1_outcome": out}, indent=1))
    print(f"[{now_iso()}] Stufe 1 fertig ({out}). Gate: F={g['F']} U={g['U']} vollstaendig={g['complete']} weiter={g['continue']}", flush=True)
    if out in ("billing", "stop") or not g["complete"] or not g["continue"]:
        print("Stopp nach Stufe 1 (Gate nicht bestanden, unvollstaendig oder abgebrochen).", flush=True)
        return 0
    while True:
        rows = R.load_rows()
        jobs = [j for j in R.stage2_jobs(rows) if not R.done(rows, j)]
        if not jobs:
            break
        out = run_queue(jobs, since)
        if out != "ok":
            print(f"[{now_iso()}] " + ("PAUSIERT: Budget-Stopp in Stufe 2, nicht finalisiert, Meldung an die Koordination." if out == "budget" else f"Stufe 2 beendet vorzeitig ({out})."), flush=True)
            return 0
    print(f"[{now_iso()}] Fertig. Verbrauch {spent_since(since):.2f} $ brutto. Auswertung: scripts/eval_recall_cg_v1.py", flush=True)
    return 0


# ---- Vorbereitung (einmalig, Modellaufrufe: S4-Auswahl je Einheit, Zielwert RLinear) -------------------------------------------------------
def prep() -> int:
    from repro_agent.loop import Orchestrator, load_claims, model_name_of
    from repro_agent.repo_source import _repo_key, find_repo_url, read_repo
    since = start_time()   # Beginn der Kostenzaehlung: Schritt 0 zaehlt in den Deckel
    os.environ.update(R.worker_env(since))
    build_overlay()
    patch_fixture_dirs()
    before = {p.relative_to(ROOT / "benchmark" / "fixtures") for p in (ROOT / "benchmark" / "fixtures").rglob("*") if p.is_file()}
    if not (OVERLAY / "2305.10721.json").exists():
        shutil.copy(RLINEAR_PAPER, OVERLAY / "2305.10721.json")   # gespeicherter Paper-Text (Tavily Extract vom 2026-10-02), kein neuer Aufruf
    for unit in R.ALL_UNITS:
        ids = R.claim_ids((unit, 0, True))
        orch = Orchestrator(claim_ids=ids, **R.ORCH_KWARGS)
        orch._tavily_getter = lambda: _forbidden_tavily
        paper, claims = load_claims(unit)
        claims = [c for c in claims if c.id in ids]
        orch._paper_id, orch._paper_title = unit, paper.get("title")
        text = json.loads((OVERLAY / f"{paper['arxiv_url'].rstrip('/').split('/abs/')[-1]}.json").read_text())["text"]
        url = find_repo_url(text)
        if unit == R.HELDOUT and url and not (OVERLAY / "repos" / f"{_repo_key(url)}_readme.json").exists():
            dump = read_repo(orch.sandbox, orch.llm, url, claims[0].text, paper["dataset"], focus_model=True, unit_key=unit,
                             model=model_name_of(claims[0].text), mode="fixture")
            readme = dump.files.get("README.md", "")
            (OVERLAY / "repos" / f"{_repo_key(url)}_readme.json").write_text(json.dumps({"url": url, "text": readme}, ensure_ascii=False, indent=1))
            print(f"{unit}: README aus dem Repo-Auszug ({len(readme)} Zeichen) statt Tavily-Extract", flush=True)
        orch.resolve_targets(paper, claims)        # Rolle 2 aus Fixture; fehlt der Zielwert, ein Modellaufruf (nur RLinear)
        orch._gather_docs(claims[0].text, paper["dataset"])   # Rolle 4: README aus Fixture, S4-Auszug je Einheit (Klon + Dateiwahl, falls Fixture fehlt)
        print(f"{unit}: fertig, fehlend {fixtures_missing(unit, ids)}", flush=True)
    after = {p.relative_to(OVERLAY) for p in OVERLAY.rglob("*") if p.is_file()}
    new = sorted(after - before)
    EXTRAS.mkdir(parents=True, exist_ok=True)
    for rel in new:
        (EXTRAS / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(OVERLAY / rel, EXTRAS / rel)
    print("Neue Fixtures (nach benchmark/experiments/recall_cg_v1_fixtures/ kopiert, committen):", *new, sep="\n  ", flush=True)
    return 0


# ---- Trockenlauf --------------------------------------------------------------------------------------------------------------------
def dry_run() -> int:
    build_overlay()
    patch_fixture_dirs()
    rows = R.load_rows()
    refs = R.reference_verdicts()
    print(f"recall cg v1: Konfiguration {R.ORCH_KWARGS}")
    print(f"Modus: {R.MODE}; Worker {R.WORKERS}; Zeitlimit je Seed {R.TRAIN_TIMEOUTS} (sonst 600); MAX_COST_PER_RUN_USD {R.MAX_COST_PER_RUN_USD}")
    print(f"Deckel {R.CAP_GROSS_USD} $ brutto, Stopp auf Aufruf-Ebene {R.STOP_GROSS_USD} $ (BUDGET_TOTAL_USD fuer Worker {R.BUDGET_TOTAL_FOR_STOP:.4f}, ABORT 95 %)")
    for unit in R.ALL_UNITS:
        miss = fixtures_missing(unit, R.claim_ids((unit, 0, True)))
        print(f"  {unit}: Claims {R.CLAIMS[unit]}, Referenz {refs[unit]['reference_mean']}, Ziel {refs[unit]['target']}; "
              + ("Fixtures vollstaendig" if not miss else "FEHLT: " + "; ".join(miss)))
    s1 = [j for j in R.stage1_jobs() if not R.done(rows, j)]
    print(f"Stufe 1: {len(s1)} offene Jobs: {[(j[0], j[1], 'P' if j[2] else '') for j in s1]}")
    est = sum(0.45 if j[2] else 0.31 for j in s1)
    print(f"  Schaetzung Stufe 1: {est:.2f} $ brutto (Plan A: 0,31 echt, 0,45 mit Placebo)")
    print("Stufe 2: RLinear Gen 0 (Placebo), 1; Entwicklungs-Gen 2 bei abweichender Klasse von Gen 0 und 1; RLinear Gen 2 bei Abweichung")
    print("Keine Aufrufe ausgefuehrt.")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if "--dry-run" in a:
        return dry_run()
    if "--prep" in a:
        return prep()
    if "--worker" in a:
        i = a.index("--worker")
        return worker(a[i + 1], int(a[i + 2]), "--placebo" in a)
    if "--mechanics" in a:
        import run_recall_cg_v1_mechanics as M
        return M.main()
    if "--run" in a:
        return run_all()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

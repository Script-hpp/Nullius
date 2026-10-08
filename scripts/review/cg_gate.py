"""Gate G0 (Genauigkeit), vorab festgelegt in benchmark/experiments/recall_cg_gate.md. Zwei Betriebsarten:

  Referenz (kein LLM):   uv run python scripts/review/cg_gate.py --ref <Einheiten|all> --out <Ausgabe> [--io-check] [--sandbox local|nebius] [--workers N]
  Gate (Token Factory):  uv run python scripts/review/cg_gate.py --run --main <Checkout mit runs/> --clones <Repo-Klone> --out <Ausgabe>
  Auswerten (kein LLM):  uv run python scripts/review/cg_gate.py --score <Ausgabe>

Das Skript wird VOR dem ersten Modell-Aufruf committet. Einheiten, Arme, Proben, Zaehlregel (`classify`/`hit`) und Entscheidungsregel
(`decide`) aendern sich danach nicht mehr. Agenten-Code laeuft nur in Docker (GateSandbox: eigene Kerngruppen, Container ohne Netz
waehrend des Trainings), nie auf dem Host. Modell-Aufrufe nur ueber LLMClient (Provider und Modelle per Umgebung, siehe das Dokument).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

REFERENCE_DIR = ROOT / "tests" / "reference"


@dataclass(frozen=True)
class Unit:
    claim_id: str
    ref_verdict: str      # Referenzurteil (benchmark/experiments/benchmark_v2.md bzw. b1_reference_rlinear.txt), vor G0 festgelegt
    ref_mean: float
    ref_file: str         # tests/reference/<Datei>
    group: str            # fail = Entwicklungseinheit mit 0 Treffern in v4 Arm B, nonreg = Nicht-Verschlechterung, new = Held-out


UNITS: dict[str, Unit] = {
    "zeng_nlinear_2023": Unit("nlinear_etth1_96_mse", "reproduziert", 0.3739, "nlinear_reference.py", "fail"),
    "wu_sgc_2019": Unit("sgc_cora_acc", "reproduziert", 0.8081, "sgc_cora_reference.py", "fail"),
    "zeng_dlinear_2023": Unit("dlinear_etth1_96_mse", "reproduziert", 0.3809, "dlinear_reference.py", "fail"),
    "velickovic_gat_2018": Unit("gat_cora_acc", "reproduziert", 0.8280, "gat_cora_reference.py", "fail"),
    "velickovic_dgi_2019": Unit("dgi_cora_acc", "teilweise", 0.8047, "dgi_cora_reference.py", "fail"),
    "shchur_gat_2018": Unit("shchur_gat_cora_acc", "reproduziert", 0.8140, "gat_lcc_reference.py", "fail"),
    "kipf_gcn_2017": Unit("gcn_cora_acc", "reproduziert", 0.8136, "gcn_cora_reference.py", "nonreg"),
    "shchur_pitfalls_2018": Unit("shchur_gcn_cora_acc", "reproduziert", 0.8107, "gcn_lcc_reference.py", "nonreg"),
    "rlinear_2023": Unit("rlinear_etth1_96_mse", "reproduziert", 0.3704, "rlinear_reference.py", "new"),
}
FAIL_UNITS = [u for u, v in UNITS.items() if v.group == "fail"]
NONREG_UNITS = [u for u, v in UNITS.items() if v.group == "nonreg"]
NEW_UNITS = [u for u, v in UNITS.items() if v.group == "new"]
ARMS = ("heute", "cg")
TEMPS = (0.2, 0.5, 0.8)
SNAPSHOT_LABEL = {"Cora": "gate-cora", "ETTh1": "gate-etth1", "Cora-LCC": "gate-cora-lcc", "CiteSeer": "gate-citeseer"}
LLM_CONCURRENCY = 8                 # gleichzeitige Modell-Aufrufe
COST_CAP_GROSS_USD = float(os.environ.get("CG_GATE_COST_CAP_GROSS_USD", "2.00"))
MIN_VALID_SHARE = 0.5               # Urteil nur, wenn mindestens die Haelfte der Seeds einen gueltigen Wert lieferte
# Entscheidungsregel (docs/recall_analysis_3.md Abschnitt 5, auf 3 Proben je Einheit und Arm angepasst; vor dem ersten Aufruf festgelegt)
RULE_A_MIN_HITS, RULE_A_MIN_GAIN = 8, 5     # (a) cg >= 8 von 18 auf den sechs Entwicklungseinheiten UND >= heute + 5
RULE_B_MAX_LOSS = 1                         # (b) Kipf + Shchur-GCN: cg >= heute - 1
RULE_C_MIN_HITS = 2                         # (c) RLinear: cg >= 2 von 3
RULE_D_MAX_MISSING = 2                      # (d) hoechstens 2 von 27 cg-Code-Aufrufen leer oder abgeschnitten


class CostStop(BaseException):
    """Kostengrenze erreicht: Lauf anhalten, Teilergebnis speichern (BaseException: loop.py faengt Exception)."""


# ---- Zaehlregel und Entscheidungsregel ---------------------------------------------------------------------------------------
@dataclass
class Sample:
    unit: str
    arm: str
    temp: float
    urteil: str | None = None        # None = noch nicht gelaufen
    mean: float | None = None
    std: float | None = None
    n_ok: int = 0
    n_seeds: int = 0
    code_missing: bool = False       # write_code lieferte nach den internen Wiederholungen keinen Code (leer oder abgeschnitten)
    attempts_seed0: int = 0
    target_reached: bool | None = None   # |Mittel - Ziel| <= Toleranz (beschreibend; die Zaehlregel ist die Klasse gleich Referenzurteil)
    seconds: float = 0.0
    note: str = ""
    facts: dict = field(default_factory=dict)


def classify(measured: list[float], n_seeds: int, verdict_fn) -> tuple[str, float | None, float | None]:
    """Klasse des Zahl-Urteils (Achse 1). Weniger als MIN_VALID_SHARE gueltige Seeds oder kein Code = 'nicht reproduzierbar (Setup)'."""
    if not measured or len(measured) < MIN_VALID_SHARE * n_seeds:
        return "nicht reproduzierbar (Setup)", None, None
    std = statistics.stdev(measured) if len(measured) > 1 else 0.0
    return verdict_fn(measured), statistics.mean(measured), std


def hit(s: Sample) -> bool:
    """Treffer = Klasse des Zahl-Urteils gleich dem vorab festgelegten Referenzurteil der Einheit (wie Benchmark v2 bis v4)."""
    return s.urteil is not None and s.urteil == UNITS[s.unit].ref_verdict


def hits(samples: list[Sample], arm: str, units: list[str]) -> int:
    return sum(1 for s in samples if s.arm == arm and s.unit in units and hit(s))


def decide(samples: list[Sample]) -> dict:
    complete = len(samples) == len(UNITS) * len(ARMS) * len(TEMPS) and all(s.urteil is not None for s in samples)
    fh, fc = hits(samples, "heute", FAIL_UNITS), hits(samples, "cg", FAIL_UNITS)
    kh, kc = hits(samples, "heute", NONREG_UNITS), hits(samples, "cg", NONREG_UNITS)
    rc = hits(samples, "cg", NEW_UNITS)
    missing = sum(1 for s in samples if s.arm == "cg" and s.code_missing)
    a = fc >= RULE_A_MIN_HITS and fc >= fh + RULE_A_MIN_GAIN
    b = kc >= kh - RULE_B_MAX_LOSS
    c = rc >= RULE_C_MIN_HITS
    d = missing <= RULE_D_MAX_MISSING
    return {"complete": complete, "a_hits_fail_heute": fh, "a_hits_fail_cg": fc, "a_pass": a,
            "b_hits_nonreg_heute": kh, "b_hits_nonreg_cg": kc, "b_pass": b,
            "c_hits_rlinear_heute": hits(samples, "heute", NEW_UNITS), "c_hits_rlinear_cg": rc, "c_pass": c,
            "d_cg_code_calls_missing": missing, "d_pass": d,
            "decision": "unvollstaendig" if not complete else ("bestanden" if a and b and c and d else "nicht bestanden")}


def per_unit_table(samples: list[Sample]) -> str:
    lines = ["| Einheit | Gruppe | Referenzurteil | heute (Treffer/3) | cg (Treffer/3) |", "|---|---|---|---|---|"]
    for u, spec in UNITS.items():
        lines.append(f"| {u} | {spec.group} | {spec.ref_verdict} | {hits(samples, 'heute', [u])} | {hits(samples, 'cg', [u])} |")
    return "\n".join(lines)


def sample_table(samples: list[Sample]) -> str:
    lines = ["| Einheit | Arm | T | Urteil | Mittel | Std | gueltige Seeds | Treffer | Ziel erreicht | Versuche Seed 0 | Dauer s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(samples, key=lambda s: (list(UNITS).index(s.unit), ARMS.index(s.arm), s.temp)):
        mean = "-" if s.mean is None else f"{s.mean:.4f}"
        std = "-" if s.std is None else f"{s.std:.4f}"
        lines.append(f"| {s.unit} | {s.arm} | {s.temp} | {s.urteil} | {mean} | {std} | {s.n_ok}/{s.n_seeds} | {'ja' if hit(s) else 'nein'} | "
                     f"{'-' if s.target_reached is None else ('ja' if s.target_reached else 'nein')} | {s.attempts_seed0} | {s.seconds:.0f} |")
    return "\n".join(lines)


# ---- Orchestrator fuer Messung und Reparatur --------------------------------------------------------------------------------
class _NoLLM:
    total_cost_usd = 0.0

    def gross(self, x):
        return x


ATTEMPT_LOG_CHARS = 1500


def log_attempt(path: Path, lock: threading.Lock, code: str, seed: int, wall_s: float, res: dict | None, exc: BaseException | None) -> None:
    """F8: Fehlerprotokoll je Versuch (Status, Sekunden, letzte 1500 Zeichen des Fehlertexts, Code-Hash, Exit-Code). Reine Beobachtung."""
    import hashlib
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "code_hash": hashlib.sha256(code.encode()).hexdigest()[:12],
           "wall_s": round(wall_s, 1)}
    if res is not None:
        row.update({"status": res.get("status"), "exit_code": res.get("exit_code"), "train_s": None if res.get("train_seconds") is None else round(res["train_seconds"], 1),
                    "detail": str(res.get("detail", ""))[-ATTEMPT_LOG_CHARS:], "soft_timeout": res.get("soft_timeout"),
                    "stack": (res.get("stack") or "")[-ATTEMPT_LOG_CHARS:] or None, "stdout_tail": (res.get("stdout_tail") or "")[-400:] or None,
                    "measured": res.get("measured")})
    if exc is not None:
        row.update({"status": "exception", "detail": f"{type(exc).__name__}: {str(exc)[-ATTEMPT_LOG_CHARS:]}"})
    with lock:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def make_orchestrator(sandbox, ask=None, llm=None, details=None, cg: bool = False, spec=None, io_helper: bool = True,
                      timeout_s: int = 600, events: list | None = None, attempt_log: Path | None = None):
    """Orchestrator ohne `run()`: nur die Teile, die Messung und Reparatur brauchen (Produktfunktionen `_attempt`, `_run_seed`,
    `_spec_conform`, `_verdict`). Produktkonfiguration wie Benchmark v4 (`graph_repair=True`, `setup_v2`); LLM-Zwischenspeicher aus."""
    from repro_agent.config import load_settings
    from repro_agent.events import EventLog
    from repro_agent.loop import Orchestrator

    events = events if events is not None else []
    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair, o.setup_v2 = details or [], True, True
    o.code_grounded, o.spec_mode, o.spec = cg, cg, (spec if cg else None)
    o.io_helper, o.train_timeout_s = io_helper, timeout_s
    o._ask = ask
    o.events = EventLog(None)
    o._emit = lambda type, **kw: events.append((type, kw))
    o._emit_live = lambda type, **kw: None
    o._announce_code = lambda *a, **k: None
    o._warn = lambda text: events.append(("warning", {"text": text}))
    o.sandbox, o._smoke_ok, o._lock, o.settings = sandbox, set(), threading.Lock(), load_settings()
    o.llm, o._cost_base = llm or _NoLLM(), 0.0
    o.tavily_mode, o.use_tavily, o.error_search_live = "fixture", False, False
    o.n_candidates = 1
    impl = [o._attempt]   # austauschbar fuer Tests (attempt.impl)
    log_lock = threading.Lock()

    def attempt(snapshot, code, seed, task, metric, **kw):   # Branch nach dem Versuch freigeben (sonst sammeln sich tausende Container)
        sandbox.begin_attempt()
        t0, res = time.time(), None
        try:
            branch, res = impl[0](snapshot, code, seed, task, metric, **kw)
            res["train_seconds"] = sandbox.times.pop(branch, None)
            return branch, res
        finally:
            sandbox.release_attempt()   # auch bei Exception (try/finally), nichts bleibt haengen
            if attempt_log is not None:   # F8
                log_attempt(attempt_log, log_lock, code, seed, time.time() - t0, res, sys.exc_info()[1])

    attempt.impl = impl
    o._attempt = attempt
    return o


def snapshot_for(sandbox, snapshots: dict, snap_lock, key: str, task):
    with snap_lock:
        if key not in snapshots:
            snapshots[key] = sandbox.base_snapshot(task.image, task.setup_cmds, label=SNAPSHOT_LABEL[key])
        return snapshots[key]


def make_sandbox(kind: str = "local"):
    """Messweg: `local` = Docker mit Kerngruppen (G0), `nebius` = Nebius-Sandbox wie im Produkt (G1)."""
    if kind == "nebius":
        from gate_sandbox import NebiusGateSandbox
        return NebiusGateSandbox()
    from gate_sandbox import GateSandbox
    return GateSandbox()


# ---- Referenzlauf (Schritt 2: Beweis des Messwegs, kein LLM) -----------------------------------------------------------------
def run_reference(units: list[str], out: Path, io_check: bool = False, sandbox_kind: str = "local", workers: int = 4) -> int:
    """tests/reference-Loesungen durch den GENAUEN Messweg: Basis-Snapshot, GateSandbox (Kerngruppen, Container ohne Netz im Training),
    Produkt-`_attempt` (statische Pruefung, Training mit `unshare -n`, Harness frisch), volle Seed-Zahl, Urteil wie der Orchestrator.
    Die Formpruefung (SMOKE=1) entfaellt: die Referenzen kennen SMOKE nicht (sie sind Selbsttest der Pipeline, nie Agenten-Code)."""
    import hashlib

    from repro_agent.loop import load_claims
    from repro_agent.tasks import TASKS

    out.mkdir(parents=True, exist_ok=True)
    sandbox, snapshots, lock = make_sandbox(sandbox_kind), {}, threading.Lock()
    rows = []
    try:
        for unit in units:
            spec = UNITS[unit]
            paper, claims = load_claims(unit)
            claim = next(c for c in claims if c.id == spec.claim_id)
            task = TASKS[paper["dataset"]]()
            snap = snapshot_for(sandbox, snapshots, lock, paper["dataset"], task)
            n_seeds = paper.get("seeds", 5)
            code = (REFERENCE_DIR / spec.ref_file).read_text()
            if spec.ref_file == "dlinear_reference.py":   # Referenzmittel 0,3809 gilt mit LR_HALVE=1 (etth1_harness_reference.txt); ohne: 0,4278
                code = code.replace('os.environ.get("LR_HALVE") == "1"', "True")
            o = make_orchestrator(sandbox.for_unit(unit), io_helper=False, timeout_s=3600)
            o._smoke_ok.add(hashlib.sha256(code.encode()).hexdigest())
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda sd: o._attempt(snap, code, sd, task, claim.metric)[1], range(n_seeds)))
            ok = [r for r in results if r["status"] == "ok"]
            vals = [r["measured"] for r in ok]
            secs = [r["train_seconds"] for r in results if r.get("train_seconds") is not None]
            allm = {n: statistics.mean(r["metrics"][n] for r in ok) for n in (ok[0]["metrics"] if ok else {})}
            verdict = o._verdict(claim, vals, n_seeds, [], (ok[0].get("baselines") if ok else None), allm) if vals else None
            row = {"unit": unit, "ref_file": spec.ref_file, "sandbox": sandbox_kind, "n_seeds": n_seeds, "n_ok": len(ok), "values": vals,
                   "mean": statistics.mean(vals) if vals else None, "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                   "urteil": verdict.urteil if verdict else None, "known_mean": spec.ref_mean, "known_verdict": spec.ref_verdict,
                   "train_seconds": secs, "max_train_seconds": max(secs) if secs else None, "wall_seconds": round(time.time() - t0),
                   "statuses": [r["status"] for r in results], "details": [str(r.get("detail", ""))[-200:] for r in results if r["status"] != "ok"]}
            rows.append(row)
            (out / "reference.jsonl").open("a").write(json.dumps(row) + "\n")
            print(f"[{unit}] {row['n_ok']}/{n_seeds} mean={row['mean']} (bekannt {spec.ref_mean}) urteil={row['urteil']} (bekannt "
                  f"{spec.ref_verdict}) max_train_s={row['max_train_seconds']} wall={row['wall_seconds']}s", flush=True)
    finally:
        sandbox.cleanup()
    return 0


# ---- Neumessung der Setup-Ausfaelle des Pilots (Schritt 2, kein LLM) --------------------------------------------------------
def run_remeasure(pilot: Path, out: Path, timeout_s: int = 600, only: str | None = None) -> int:
    """Die 9 Setup-Ausfaelle des Pilots (recall_cg_pilot_samples) mit unveraendertem Code durch den neuen Messweg: Kerngruppen-Pool,
    Prozesse nach Timeout beendet, Container ohne Netz im Training. Kein Helfer (der Code kennt ihn nicht), keine Reparatur, kein LLM.
    Endgueltiger Code der Probe = Reparatur, falls vorhanden (der Pilot ersetzt den Code, wenn der alte nicht lief), sonst Original."""
    from repro_agent.dossier import Dossier
    from repro_agent.loop import MAX_PARALLEL, load_claims
    from repro_agent.tasks import TASKS

    pilot_units = {"zeng_nlinear_2023": "nlinear_etth1_96_mse", "wu_sgc_2019": "sgc_cora_acc", "velickovic_gat_2018": "gat_cora_acc", "kipf_gcn_2017": "gcn_cora_acc"}
    out.mkdir(parents=True, exist_ok=True)
    sandbox, snapshots, lock = make_sandbox("local"), {}, threading.Lock()
    jobs = []
    for unit, claim_id in pilot_units.items():
        for f in sorted((pilot / unit).glob("final_*.json")):
            d = json.loads(f.read_text())
            if d["urteil"] != "nicht reproduzierbar (Setup)":
                continue
            arm, t = d["arm"], d["temp"]
            if only and f"{unit}:{arm}:{t}" not in only.split(","):
                continue
            repair = pilot / unit / f"repair_{arm}_{t}.py"
            code_file = repair if repair.exists() else pilot / unit / f"code_{arm}_{t}.py"
            if code_file.exists():
                jobs.append((unit, arm, t, code_file, d))

    def one(job):
        unit, arm, t, code_file, d = job
        paper, claims = load_claims(unit)
        claim = next(c for c in claims if c.id == pilot_units[unit])
        task = TASKS[paper["dataset"]]()
        snap = snapshot_for(sandbox, snapshots, lock, paper["dataset"], task)
        n_seeds = paper.get("seeds", 5)
        code = code_file.read_text()
        o = make_orchestrator(sandbox.for_unit(unit), io_helper=False, timeout_s=timeout_s)
        t0 = time.time()
        branch, res0 = o._attempt(snap, code, 0, task, claim.metric)
        measured, allm, baselines = [], {}, res0.get("baselines")
        if res0["status"] == "ok":
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
                rest = list(pool.map(lambda sd: o._attempt(snap, code, sd, task, claim.metric)[1], range(1, n_seeds)))
            allr = [res0] + [r for r in rest if r["status"] == "ok"]
            measured = [r["measured"] for r in allr]
            allm = {n: statistics.mean(r["metrics"][n] for r in allr) for n in allr[0]["metrics"]}
        urteil, mean, std = classify(measured, n_seeds, lambda m: o._verdict(claim, m, n_seeds, [], baselines, allm).urteil)
        row = {"unit": unit, "arm": arm, "temp": t, "code_file": code_file.name, "as_run": d["urteil"], "as_run_mean": d["mean"],
               "re_urteil": urteil, "re_mean": mean, "re_std": std, "n_ok": len(measured), "n_seeds": n_seeds,
               "seed0_status": res0["status"], "seed0_detail": str(res0.get("detail", ""))[-300:], "seed0_train_s": res0.get("train_seconds"),
               "seconds": round(time.time() - t0), "hit_as_run": d["urteil"] == UNITS[unit].ref_verdict, "hit_re": urteil == UNITS[unit].ref_verdict}
        (out / "remeasure.jsonl").open("a").write(json.dumps(row) + "\n")
        print(f"[{unit}] {arm} T={t}: as run {d['urteil']} -> re-measured {urteil} mean={mean} n_ok={len(measured)}/{n_seeds} seed0={res0['status']} ({row['seconds']}s)", flush=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(one, jobs))
    finally:
        sandbox.cleanup()
    return 0


# ---- Gate-Lauf ----------------------------------------------------------------------------------------------------------------
class Spend:
    """Kosten des Gates ueber alle Prozesse und Wiederaufnahmen (out/spend.jsonl, brutto), harte Grenze auf Aufruf-Ebene."""

    def __init__(self, path: Path, llm, cap: float):
        self.path, self.llm, self.cap, self.lock = path, llm, cap, threading.Lock()

    def total(self) -> float:
        return sum(json.loads(line)["gross"] for line in self.path.open()) if self.path.exists() else 0.0

    def check(self) -> None:
        with self.lock:
            if self.total() >= self.cap:
                raise CostStop(f"Kostengrenze {self.cap} $ brutto erreicht ({self.total():.4f} $)")

    def add(self, role: str, net: float) -> None:
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "net": net, "gross": self.llm.gross(net)}) + "\n")


def make_ask(llm, spend: Spend):
    sem = threading.Semaphore(LLM_CONCURRENCY)

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        spend.check()
        with sem:
            spend.check()
            resp = llm.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                temperature=temperature, max_tokens=max_tokens)
        spend.add(str(role), resp.cost_usd)
        return resp

    return ask


def _spec_from_json(d: dict):
    from repro_agent.spec import Spec, SpecItem
    items = [SpecItem(**i) for i in d.pop("items")]
    return Spec(items=items, **d)


def run_gate(main: Path, clones: Path, out: Path, units: list[str], timeouts: dict[str, int], jobs: int = 18) -> list[Sample]:
    from dataclasses import asdict as _asdict

    from repro_agent.llm import LLMClient
    from repro_agent.spec import build_spec
    from repro_agent.tasks import TASKS
    from uptake_probe import FACTS, check_facts, load_unit

    out.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    spend = Spend(out / "spend.jsonl", llm, COST_CAP_GROSS_USD)
    ask = make_ask(llm, spend)
    sandbox, snapshots, snap_lock = make_sandbox("local"), {}, threading.Lock()
    samples: list[Sample] = []
    samples_lock = threading.Lock()

    def sample_file(unit, arm, t):
        return out / unit / f"final_{arm}_{t}.json"

    def unit_arm(unit: str, arm: str, data: dict, task, claim, spec, snap) -> None:
        ud = out / unit
        plan_file = ud / f"plan_{arm}.md"
        orch = make_orchestrator(sandbox.for_unit(unit), ask=ask, llm=llm, details=data["details"], cg=arm == "cg", spec=spec,
                                 timeout_s=timeouts.get(unit, 600))
        if not plan_file.exists():
            plan_file.write_text(orch.plan(data["paper"], claim, task))
        plan = plan_file.read_text()

        def one(t: float) -> None:
            final = sample_file(unit, arm, t)
            n_seeds = data["paper"].get("seeds", 5)
            if final.exists():
                with samples_lock:
                    samples.append(Sample(**json.loads(final.read_text())))
                return
            s = Sample(unit, arm, t, n_seeds=n_seeds)
            code_file, none_file = ud / f"code_{arm}_{t}.py", ud / f"code_{arm}_{t}.none"
            t0 = time.time()
            if not code_file.exists() and not none_file.exists():
                try:
                    code_file.write_text(orch.write_code(plan, task, temperature=t))
                except (ValueError, RuntimeError) as exc:
                    none_file.write_text(str(exc)[:300])
            if not code_file.exists():
                s.code_missing, s.urteil, s.note = True, "nicht reproduzierbar (Setup)", "kein Code vom Modell: " + none_file.read_text()
            else:
                measure_sample(orch, s, code_file.read_text(), data, task, claim, snap, plan, ud / f"run_{arm}_{t}", n_seeds)
                if unit in FACTS:
                    s.facts = check_facts(unit, (ud / f"final_code_{arm}_{t}.py").read_text() if (ud / f"final_code_{arm}_{t}.py").exists() else code_file.read_text())
            s.seconds = round(time.time() - t0, 1)
            final.write_text(json.dumps(asdict(s), default=str))
            with samples_lock:
                samples.append(s)
            print(f"[{unit}] {arm} T={t}: {s.urteil} mean={s.mean} n_ok={s.n_ok}/{s.n_seeds} hit={hit(s)} ({s.seconds:.0f}s)", flush=True)

        with ThreadPoolExecutor(max_workers=len(TEMPS)) as pool:
            list(pool.map(one, TEMPS))

    def unit_job(unit: str):
        data = load_unit(main, clones, unit)
        task = TASKS[data["paper"]["dataset"]]()
        claim = data["claim"]
        assert claim.id == UNITS[unit].claim_id, (claim.id, unit)
        snap = snapshot_for(sandbox, snapshots, snap_lock, data["paper"]["dataset"], task)
        ud = out / unit
        ud.mkdir(exist_ok=True)
        spec_file = ud / "spec.json"
        if spec_file.exists():
            spec = _spec_from_json(json.loads(spec_file.read_text()))
        else:
            spec = build_spec(lambda s, u: ask("extract", s, u, max_tokens=12000, temperature=0.0).content,
                              data["details"], data["docs"], claim.text, data["paper"]["dataset"], numeric_only=True)
            spec_file.write_text(json.dumps(_asdict(spec), indent=1, default=str))
        with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
            list(pool.map(lambda arm: unit_arm(unit, arm, data, task, claim, spec, snap), ARMS))

    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(unit_job, u) for u in units]
            for f in futures:
                f.result()
    finally:
        sandbox.cleanup()
        store(samples, out)
    return samples


def measure_sample(orch, s: Sample, code: str, data: dict, task, claim, snap: str, plan: str, run_dir: Path, n_seeds: int) -> None:
    """Produktpfad fuer eine Probe: CONFIG-Befunde (nur cg, `_spec_conform`, hoechstens MAX_SPEC_REPAIRS Reparaturen), Seed 0 mit
    Reparaturschleife (`_run_seed`, hoechstens MAX_ATTEMPTS_PER_CLAIM Versuche), die uebrigen Seeds parallel, Urteil wie der Orchestrator.
    Nicht enthalten (beide Arme gleich): Kandidatenauswahl, Fehler- und Graph-Reparatur nach Seed 0, Repo-Abgleich."""
    from repro_agent.dossier import Dossier
    from repro_agent.loop import MAX_PARALLEL

    run_dir.mkdir(parents=True, exist_ok=True)
    dossier = Dossier(paper_id=s.unit)
    try:
        code, _, _ = orch._spec_conform(claim, None, plan, task, code)
    except (ValueError, RuntimeError) as exc:
        s.note += f"Steckbrief-Pruefung/Reparatur ohne Code: {str(exc)[:150]}; "
    v0, note0, code = orch._run_seed(claim, task, snap, dossier, run_dir, plan, code, 0)
    s.attempts_seed0 = sum(1 for a in dossier.attempts if a.seed == 0)
    (run_dir.parent / f"final_code_{s.arm}_{s.temp}.py").write_text(code)
    notes = [] if note0 is None else [note0]
    measured: list[float] = []
    if v0 is not None:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            rest = list(pool.map(lambda sd: orch._run_seed(claim, task, snap, dossier, run_dir, plan, code, sd), range(1, n_seeds)))
        measured = [v0] + [v for v, _, _ in rest if v is not None]
        notes += [n for _, n, _ in rest if n]
    per_metric: dict[str, list[float]] = {}
    for a in dossier.attempts:
        if a.metrics and a.measured is not None:
            for name, value in a.metrics.items():
                per_metric.setdefault(name, []).append(value)
    all_metrics = {n: statistics.mean(v) for n, v in per_metric.items()}

    def verdict_fn(m):
        return orch._verdict(claim, m, n_seeds, notes, dossier.baselines, all_metrics).urteil

    s.urteil, s.mean, s.std = classify(measured, n_seeds, verdict_fn)
    s.n_ok = len(measured)
    if s.mean is not None:
        s.target_reached = abs(s.mean - claim.target) <= claim.tolerance
    s.note += "; ".join(n[:200] for n in notes)


def store(samples: list[Sample], out: Path) -> None:
    with open(out / "results.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), default=str) + "\n")


def report(samples: list[Sample], spent: float | None = None) -> str:
    lines = [sample_table(samples), "", per_unit_table(samples), "", "```json", json.dumps(decide(samples), indent=1, ensure_ascii=False), "```"]
    if spent is not None:
        lines.append(f"Kosten (brutto, Log dieses Gates): {spent:.4f} $")
    return "\n".join(lines)


def load_timeouts(path: Path | None) -> dict[str, int]:
    return json.loads(path.read_text()) if path and path.exists() else {}


def main() -> int:
    argv = sys.argv[1:]
    arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    if "--score" in argv:
        out = Path(arg("--score"))
        samples = [Sample(**json.loads(line)) for line in (out / "results.jsonl").read_text().splitlines()]
        spent = Spend(out / "spend.jsonl", type("L", (), {"gross": staticmethod(lambda x: x)})(), 1e9).total() if (out / "spend.jsonl").exists() else None
        print(report(samples, spent))
        return 0
    if "--ref" in argv:
        sel = arg("--ref")
        units = list(UNITS) if sel == "all" else sel.split(",")
        return run_reference(units, Path(arg("--out")), "--io-check" in argv, sandbox_kind=arg("--sandbox", "local"), workers=int(arg("--workers", "4")))
    if "--remeasure" in argv:
        return run_remeasure(Path(arg("--remeasure")), Path(arg("--out")), only=arg("--only"))
    if "--run" in argv:
        units = arg("--units").split(",") if "--units" in argv else list(UNITS)
        try:
            samples = run_gate(Path(arg("--main")), Path(arg("--clones")), Path(arg("--out")), units, load_timeouts(Path(arg("--timeouts")) if arg("--timeouts") else None))
        except CostStop as exc:
            print(f"STOPP: {exc}. Wiederaufnahme mit demselben Befehl (out/ bleibt).", file=sys.stderr, flush=True)
            return 2
        print(report(samples))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())

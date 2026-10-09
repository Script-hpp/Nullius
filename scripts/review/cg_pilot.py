"""Code-grounded Pilot (Genauigkeit), vorab festgelegt in benchmark/experiments/recall_cg_pilot.md.

Dieses Skript wird VOR dem ersten Modell-Aufruf committet. Einheiten, Arme, Proben, Zaehlregel (`classify`) und Entscheidungsregel
(`decide`) aendern sich danach nicht mehr. Es ist ein In-Sample-Pilot, keine Benchmark-Zahl.

  uv run python scripts/review/cg_pilot.py --main <Checkout mit runs/> --clones <Verzeichnis mit Repo-Klonen> --out <Ausgabe>
  uv run python scripts/review/cg_pilot.py --score <Ausgabe>        (nur auswerten, kein LLM, kein Docker)

Modell-Aufrufe nur ueber LLMClient (Prototyp: LLM_PROVIDER=openrouter, kostenlos), jede HTTP-Anfrage wird gezaehlt (auch 429 und
Wiederholungen), harte Grenze CALL_CAP ueber alle Laeufe/Tage (out/calls.jsonl). Kein Tavily, kein Nebius, kein Token Factory.
Agenten-Code laeuft nur in Docker (LocalSandbox, Container ohne Netz), nie auf dem Host.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

# unit -> (Claim-Id, Referenzurteil, Referenzmittel); Referenzurteile aus benchmark/experiments/benchmark_v2.md
UNITS = {
    "zeng_nlinear_2023": ("nlinear_etth1_96_mse", "reproduziert", 0.3739),
    "wu_sgc_2019": ("sgc_cora_acc", "reproduziert", 0.8081),
    "velickovic_gat_2018": ("gat_cora_acc", "reproduziert", 0.8280),
    "kipf_gcn_2017": ("gcn_cora_acc", "reproduziert", 0.8136),
}
FAIL_UNITS = ["zeng_nlinear_2023", "wu_sgc_2019", "velickovic_gat_2018"]   # in v4 Arm B 0/3 (Entwicklungseinheiten)
NONREG_UNITS = ["kipf_gcn_2017"]
ARMS = ("heute", "cg")
TEMPS = (0.2, 0.8)
CALL_CAP = 45            # HTTP-Anfragen insgesamt (S1 + Plaene + Code + Reparaturen + Wiederholungen + 429)
MIN_INTERVAL_S = 3.3     # Free-Tier: 20 Anfragen pro Minute
MIN_VALID_SHARE = 0.5    # Urteil nur, wenn mindestens die Haelfte der Seeds einen gueltigen Wert lieferte
MAX_REPAIRS = 1          # je Probe, gemeinsame Reparatur (Fehler beim Seed-0-Lauf, bei cg zusaetzlich CONFIG-Befunde)
CPU_SLOTS, CORES_PER_SLOT = 4, 4   # 4 parallele Seeds auf 16 Kernen (Infrastruktur-Korrektur, siehe Addendum 2)
PASS_FAIL_UNITS_MIN, PASS_FAIL_UNITS_GAIN, PASS_NONREG_LOSS = 3, 2, 1


class QuotaStop(BaseException):
    """Aufrufgrenze oder Tageslimit: Lauf anhalten, Teilergebnis speichern (BaseException: loop.py faengt Exception)."""


# ---- Zaehlregel und Entscheidungsregel ---------------------------------------------------------------------------------------
@dataclass
class Result:
    unit: str
    arm: str
    temp: float
    urteil: str | None = None       # None = nicht gelaufen (Grenze erreicht)
    mean: float | None = None
    std: float | None = None
    n_ok: int = 0
    n_seeds: int = 0
    note: str = ""
    facts: dict = field(default_factory=dict)
    repaired: bool = False
    overrides: int = 0


def classify(measured: list[float], n_seeds: int, verdict_fn) -> tuple[str, float | None, float | None]:
    """Klasse des Zahl-Urteils (Achse 1). Weniger als MIN_VALID_SHARE gueltige Seeds oder kein Code = 'nicht reproduzierbar (Setup)'.
    verdict_fn(measured) -> Urteil des Orchestrators (judge + Agenten-Fehler-Pruefung)."""
    if not measured or len(measured) < MIN_VALID_SHARE * n_seeds:
        return "nicht reproduzierbar (Setup)", None, None
    std = statistics.stdev(measured) if len(measured) > 1 else 0.0
    return verdict_fn(measured), statistics.mean(measured), std


def hit(r: Result) -> bool:
    return r.urteil is not None and r.urteil == UNITS[r.unit][1]


def hits(results: list[Result], arm: str, units: list[str]) -> int:
    return sum(1 for r in results if r.arm == arm and r.unit in units and hit(r))


def decide(results: list[Result]) -> dict:
    """Vorab festgelegt: bestanden, wenn cg auf den 3 Entwicklungseinheiten >= 3 von 6 trifft UND >= heute + 2, und Kipf cg nicht
    schlechter als heute - 1. Nur bei vollstaendigem Pilot (alle 16 Proben gelaufen); sonst 'unvollstaendig'."""
    complete = len(results) == len(UNITS) * len(ARMS) * len(TEMPS) and all(r.urteil is not None for r in results)
    hf_h, hf_c = hits(results, "heute", FAIL_UNITS), hits(results, "cg", FAIL_UNITS)
    hk_h, hk_c = hits(results, "heute", NONREG_UNITS), hits(results, "cg", NONREG_UNITS)
    c1 = hf_c >= PASS_FAIL_UNITS_MIN and hf_c >= hf_h + PASS_FAIL_UNITS_GAIN
    c2 = hk_c >= hk_h - PASS_NONREG_LOSS
    return {"complete": complete, "hits_fail_heute": hf_h, "hits_fail_cg": hf_c, "hits_kipf_heute": hk_h, "hits_kipf_cg": hk_c,
            "c1_cg_on_failing_units": c1, "c2_kipf_not_worse": c2,
            "decision": "unvollstaendig" if not complete else ("bestanden" if c1 and c2 else "nicht bestanden")}


# ---- Aufruf-Zaehler (HTTP-Ebene) ---------------------------------------------------------------------------------------------
class CallCounter:
    def __init__(self, path: Path, cap: int = CALL_CAP):
        self.path, self.cap, self.lock, self.last = path, cap, threading.Lock(), 0.0

    def count(self) -> int:
        return sum(1 for _ in self.path.open()) if self.path.exists() else 0

    def wrap(self, client) -> None:
        orig = client.post

        def post(*a, **k):
            with self.lock:
                n = self.count()
                if n >= self.cap:
                    raise QuotaStop(f"Aufrufgrenze {self.cap} erreicht")
                wait = MIN_INTERVAL_S - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                with self.path.open("a") as f:
                    f.write(json.dumps({"n": n + 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            resp = orig(*a, **k)
            if resp.status_code == 429 and any(w in resp.text.lower() for w in ("per-day", "per day", "daily")):
                raise QuotaStop(f"Tageslimit des Anbieters: {resp.text[:200]}")
            return resp

        client.post = post


# ---- Sandbox ohne Netz -------------------------------------------------------------------------------------------------------
class NoNetLocalSandbox:
    """LocalSandbox (Docker). Waehrend des Agenten-Codes ist der Container vom Netz getrennt (`docker network disconnect bridge`),
    fuer den Harness (laedt Testlabels frisch aus dem Original-Datensatz) wieder verbunden. `unshare -n` entfaellt (braucht Rechte
    im Container). Snapshot-Bau (Daten, pip) laeuft mit Netz wie im Produkt."""

    NET = "bridge"

    def __init__(self):
        from repro_agent.sandbox.local import LocalSandbox
        self.inner, self.branches, self._n, self._lock = LocalSandbox(), [], 0, threading.Lock()

    def base_snapshot(self, image, setup_cmds, label=None):
        return self.inner.base_snapshot(image, setup_cmds, label=label)

    def branch(self, snapshot):
        import uuid
        name = f"cgpilot-{uuid.uuid4().hex[:8]}"
        with self._lock:   # Infrastruktur-Korrektur: jeder Container bekommt eigene Kerne (4 parallele Seeds ueberbuchten sonst alle 16 Kerne)
            slot, self._n = self._n % CPU_SLOTS, self._n + 1
        cores = f"{slot * CORES_PER_SLOT}-{slot * CORES_PER_SLOT + CORES_PER_SLOT - 1}"
        subprocess.run(["docker", "run", "-d", "--name", name, "--cpuset-cpus", cores, snapshot, "sleep", "infinity"], check=True,
                       capture_output=True)
        subprocess.run(["docker", "exec", name, "mkdir", "-p", "/work", "/output"], check=True, capture_output=True)
        self.branches.append(name)
        return name

    def run(self, branch, cmd, timeout_s):
        if "unshare -n " not in cmd:
            return self.inner.run(branch, cmd, timeout_s)
        subprocess.run(["docker", "network", "disconnect", self.NET, branch], check=True, capture_output=True)
        try:
            return self.inner.run(branch, cmd.replace("unshare -n ", ""), timeout_s)
        finally:
            subprocess.run(["docker", "network", "connect", self.NET, branch], capture_output=True)

    def put_file(self, branch, path, content):   # LocalSandbox.put_file legt das Zielverzeichnis nicht an
        subprocess.run(["docker", "exec", branch, "mkdir", "-p", str(Path(path).parent)], check=True, capture_output=True)
        self.inner.put_file(branch, path, content)

    def put_files(self, branch, files):
        for path, content in files.items():
            self.put_file(branch, path, content)

    def get_file(self, branch, path):
        return self.inner.get_file(branch, path)

    def cleanup(self):
        for name in self.branches:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        self.branches.clear()


# ---- Lauf --------------------------------------------------------------------------------------------------------------------
def _spec_from_json(d: dict):
    from repro_agent.spec import Spec, SpecItem
    items = [SpecItem(**i) for i in d.pop("items")]
    return Spec(items=items, **d)


def _spec_to_json(spec) -> dict:
    from dataclasses import asdict
    return asdict(spec)   # inkl. excerpt (roher Ausschnitt), fuer Wiederaufnahme und Beleg


def make_orchestrator(ask, data: dict, cg: bool, spec, sandbox, events: list):
    import threading as th

    from repro_agent.config import load_settings
    from repro_agent.events import EventLog
    from repro_agent.loop import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair, o.setup_v2 = data["details"], True, True   # Produktkonfiguration wie Benchmark v4
    o.code_grounded, o.spec_mode, o.spec = cg, cg, (spec if cg else None)
    o._ask = ask
    o.events = EventLog(None)
    o._emit = lambda type, **kw: events.append((type, kw))
    o._announce_code = lambda *a, **k: None
    o._warn = lambda text: events.append(("warning", {"text": text}))
    o.sandbox, o._smoke_ok, o._lock, o.settings = sandbox, set(), th.Lock(), load_settings()
    o.tavily_mode, o.use_tavily = "fixture", False
    return o


def run_pilot(ask, counter: CallCounter, main: Path, clones: Path, out: Path, units: list[str]) -> list[Result]:
    from repro_agent.spec import build_spec
    from repro_agent.tasks import TASKS
    from uptake_probe import load_unit

    out.mkdir(parents=True, exist_ok=True)
    sandbox = NoNetLocalSandbox()
    snapshots: dict[str, str] = {}
    results: list[Result] = []
    try:
        # Stufe 1: Generierung (S1 nur cg, Plaene, Code) je Einheit; danach Messung. Wiederaufnahme ueber Dateien in out/.
        prepared = {}
        for unit in units:
            data = load_unit(main, clones, unit)
            task = TASKS[data["paper"]["dataset"]]()
            ud = out / unit
            ud.mkdir(exist_ok=True)
            claim = data["claim"]   # echter Claim (claims[0])
            assert claim.id == UNITS[unit][0], (claim.id, unit)
            spec_file = ud / "spec.json"
            if spec_file.exists():
                spec = _spec_from_json(json.loads(spec_file.read_text()))
            else:
                spec = build_spec(lambda s, u: ask("extract", s, u, max_tokens=12000, temperature=0.0).content,
                                  data["details"], data["docs"], claim.text, data["paper"]["dataset"], numeric_only=True)
                spec_file.write_text(json.dumps(_spec_to_json(spec), indent=1, default=str))
            events: list = []
            orch = {arm: make_orchestrator(ask, data, arm == "cg", spec, sandbox, events) for arm in ARMS}
            plans = {}
            for arm in ARMS:
                pf = ud / f"plan_{arm}.md"
                if not pf.exists():
                    pf.write_text(orch[arm].plan(data["paper"], claim, task))
                plans[arm] = pf.read_text()
            for arm in ARMS:
                for t in TEMPS:
                    cf = ud / f"code_{arm}_{t}.py"
                    if cf.exists():
                        continue
                    try:
                        cf.write_text(orch[arm].write_code(plans[arm], task, temperature=t))
                    except (ValueError, RuntimeError) as exc:
                        (ud / f"code_{arm}_{t}.none").write_text(str(exc)[:300])
            prepared[unit] = (data, task, claim, spec, orch, plans, events)
        # Stufe 2: Messung (Docker), Reparaturen abwechselnd heute/cg, damit eine Aufrufgrenze beide Arme gleich trifft
        for unit in units:
            data, task, claim, spec, orch, plans, events = prepared[unit]
            results += measure_unit(unit, data, task, claim, spec, orch, plans, events, out / unit, sandbox, snapshots)
    finally:
        sandbox.cleanup()
        store(results, out)
    return results


def measure_unit(unit, data, task, claim, spec, orch, plans, events, ud: Path, sandbox, snapshots) -> list[Result]:
    from repro_agent import spec_check
    from repro_agent.harness.code_check import find_violations  # noqa: F401  (statische Pruefung laeuft in _run_and_eval)
    from uptake_probe import FACTS, check_facts

    n_seeds = data["paper"].get("seeds", 5)
    key = data["paper"]["dataset"]
    if key not in snapshots:
        snapshots[key] = sandbox.base_snapshot(task.image, task.setup_cmds, label=f"cgpilot-{key.lower()}")
    snap = snapshots[key]
    samples = []
    for arm in ARMS:
        for t in TEMPS:
            samples.append({"arm": arm, "t": t, "file": ud / f"code_{arm}_{t}.py", "final": ud / f"final_{arm}_{t}.json"})
    done = [json.loads(s["final"].read_text()) for s in samples if s["final"].exists()]
    todo = [s for s in samples if not s["final"].exists()]
    results = [Result(**d) for d in done]
    # Seed 0 je Probe (inkl. Formpruefung); Befund -> eine gemeinsame Reparatur
    state = {}
    for s in todo:
        if not s["file"].exists():
            r = Result(unit, s["arm"], s["t"], "nicht reproduzierbar (Setup)", None, None, 0, n_seeds, "kein Code vom Modell")
            s["final"].write_text(json.dumps(r.__dict__, default=str))
            results.append(r)
            continue
        code = s["file"].read_text()
        o = orch[s["arm"]]
        branch, res0 = o._attempt(snap, code, 0, task, claim.metric)
        findings = []
        if res0["status"] != "ok":
            findings.append(("run", o.triage(res0["detail"])))
        if s["arm"] == "cg" and spec.enforced():
            _, rep = spec_check.config_check(code, spec)
            fails = spec_check.failures(rep, [])
            if fails:
                findings.append(("spec", spec_check.repair_message(fails, spec)))
        state[(s["arm"], s["t"])] = {"code": code, "res0": res0, "findings": findings, "s": s}
    # Reparaturen abwechselnd
    order = [k for k in sorted(state, key=lambda k: (k[1], ARMS.index(k[0]))) if state[k]["findings"]]
    for k in order:
        st = state[k]
        o = orch[k[0]]
        msg = "\n\n".join(m for _, m in st["findings"])
        rf = ud / f"repair_{k[0]}_{k[1]}.py"
        try:
            new = rf.read_text() if rf.exists() else o.repair_code(plans[k[0]], task, st["code"], msg)   # Wiederaufnahme: keine zweite Anfrage
        except (ValueError, RuntimeError) as exc:
            st["note"] = f"Reparatur ohne Code: {str(exc)[:150]}"
            continue
        rf.write_text(new)
        branch, res_new = o._attempt(snap, new, 0, task, claim.metric)
        st["repaired"] = True
        if res_new["status"] == "ok" or st["res0"]["status"] != "ok":
            st.update(code=new, res0=res_new)   # Reparatur ersetzt den Code; war der alte lauffaehig und der neue nicht, bleibt der alte
    # alle Seeds
    for k, st in state.items():
        o = orch[k[0]]
        code, res0 = st["code"], st["res0"]
        measured, baselines, allm = [], None, {}
        if res0["status"] == "ok":
            measured = [res0["measured"]]
            baselines = res0.get("baselines")
            metrics = [res0.get("metrics") or {}]

            def one(sd):
                return o._attempt(snap, code, sd, task, claim.metric)[1]

            with ThreadPoolExecutor(max_workers=4) as pool:
                rest = list(pool.map(one, range(1, n_seeds)))
            for rr in rest:
                if rr["status"] == "ok":
                    measured.append(rr["measured"])
                    metrics.append(rr.get("metrics") or {})
            names = {n for m in metrics for n in m}
            allm = {n: statistics.mean(m[n] for m in metrics if n in m) for n in names}
        note = st.get("note", "") or ("; ".join(m[:200] for _, m in st["findings"]) if st["findings"] else "")
        if res0["status"] != "ok":
            note += f" | letzter Fehler: {str(res0.get('detail', ''))[-200:]}"

        def verdict_fn(m):
            return o._verdict(claim, m, n_seeds, [], baselines, allm).urteil

        urteil, mean, std = classify(measured, n_seeds, verdict_fn)
        ovr = code.count("# spec override")   # S2-Overrides im endgueltigen Code
        r = Result(unit, k[0], k[1], urteil, mean, std, len(measured), n_seeds, note.strip(),
                   check_facts(unit, code) if unit in FACTS else {}, st.get("repaired", False), ovr)
        st["s"]["final"].write_text(json.dumps(r.__dict__, default=str))
        results.append(r)
        print(f"[{unit}] {k[0]} T={k[1]}: {urteil} mean={mean} n_ok={len(measured)}/{n_seeds}", flush=True)
    sandbox.cleanup()
    return results


def store(results: list[Result], out: Path) -> None:
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, default=str) + "\n")


def report(results: list[Result], calls: int | None = None) -> str:
    lines = ["| Einheit | Arm | T | Urteil | Mittel | Std | gueltige Seeds | Treffer | Reparatur | Fakten |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (list(UNITS).index(r.unit), ARMS.index(r.arm), r.temp)):
        facts = ", ".join(f"{k}={'ja' if v else 'nein'}" for k, v in r.facts.items())
        mean = "-" if r.mean is None else f"{r.mean:.4f}"
        std = "-" if r.std is None else f"{r.std:.4f}"
        lines.append(f"| {r.unit} | {r.arm} | {r.temp} | {r.urteil} | {mean} | {std} | {r.n_ok}/{r.n_seeds} | "
                     f"{'ja' if hit(r) else 'nein'} | {'ja' if r.repaired else 'nein'} | {facts} |")
    lines += ["", "```json", json.dumps(decide(results), indent=1, ensure_ascii=False) + "```"]
    if calls is not None:
        lines.append(f"HTTP-Anfragen gezaehlt: {calls} von {CALL_CAP}")
    return "\n".join(lines)


def main() -> int:
    argv = sys.argv[1:]
    arg = lambda n: argv[argv.index(n) + 1]
    if "--score" in argv:
        out = Path(arg("--score"))
        results = [Result(**json.loads(line)) for line in (out / "results.jsonl").read_text().splitlines()]
        calls = CallCounter(out / "calls.jsonl").count()
        print(report(results, calls))
        return 0
    from repro_agent.llm import LLMClient
    llm = LLMClient()
    out = Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    counter = CallCounter(out / "calls.jsonl")
    counter.wrap(llm._client)

    import os
    cost_cap = float(os.environ.get("CG_PILOT_COST_CAP_GROSS_USD", "0"))   # Addendum: Token Factory, harte Grenze brutto fuer den Rest

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        if cost_cap and llm.gross(llm.total_cost_usd) >= cost_cap:
            raise QuotaStop(f"Kostengrenze {cost_cap} $ brutto erreicht ({llm.gross(llm.total_cost_usd):.4f} $)")
        return llm.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            temperature=temperature, max_tokens=max_tokens)

    units = arg("--units").split(",") if "--units" in argv else list(UNITS)
    try:
        results = run_pilot(ask, counter, Path(arg("--main")), Path(arg("--clones")), out, units)
    except QuotaStop as exc:
        print(f"STOPP: {exc}. Wiederaufnahme mit demselben Befehl (out/ bleibt).", file=sys.stderr, flush=True)
        return 2
    print(report(results, counter.count()))
    print(f"Kosten (netto, Log): {llm.total_cost_usd:.4f} $")
    return 0


if __name__ == "__main__":
    sys.exit(main())

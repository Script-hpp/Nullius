"""Gate G1 (Genauigkeit), vorab festgelegt in benchmark/experiments/recall_cg_gate_g1.md. Zweiter und einziger weiterer Versuch nach G0.

  Offline-Pruefung (kein LLM):  uv run python scripts/review/cg_gate_g1.py --audit --g0 <G0-Ausgabe> --main <Checkout mit runs/> --out <Datei.md>
  Gate (Token Factory):         uv run python scripts/review/cg_gate_g1.py --run --g0 <G0-Ausgabe> --main <Checkout mit runs/> --out <Ausgabe> --timeouts <json>
  Auswerten (kein LLM):         uv run python scripts/review/cg_gate_g1.py --score <Ausgabe> --g0 <G0-Ausgabe>

Dieses Skript wird VOR dem ersten Modell-Aufruf committet. Einheiten, Arme, Proben, Zaehlregel (`hit` aus cg_gate.py) und Entscheidungsregel
(`decide_g1`) aendern sich danach nicht mehr. Messweg: Nebius-Sandbox (`NebiusGateSandbox`), Netzsperre `unshare -n` und Harness wie im Produkt.
Agenten-Code laeuft nie auf dem Host. Modell-Aufrufe nur ueber LLMClient. Kein Tavily-Aufruf, keine Klone (Steckbrief wird aus G0 ohne Extraktor-Aufruf
neu gerechnet, `spec.rebuild_spec`).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

import cg_gate as g0   # noqa: E402  (G0-Regeln, Messung und Probe-Typ unveraendert)
from cg_gate import ARMS, TEMPS, UNITS, CostStop, Sample, Spend, hit  # noqa: E402

G1_GAT = ["velickovic_gat_2018", "shchur_gat_2018"]          # beide Arme, Regel (a)
G1_NONREG = ["shchur_pitfalls_2018", "rlinear_2023"]         # nur Arm cg (Nichtverschlechterung, F4 beruehrt sie)
G1_CELLS = [(u, a) for u in G1_GAT for a in ARMS] + [(u, "cg") for u in G1_NONREG]
DEV_UNITS = g0.FAIL_UNITS                                    # die sechs Entwicklungseinheiten der Regel (a)
COST_CAP_GROSS_USD = float(os.environ.get("CG_GATE_G1_COST_CAP_GROSS_USD", "1.00"))
NONREG_MIN_HITS = 2                                          # (b') je Nichtverschlechterungseinheit: cg >= 2 von 3 (G0: 3 von 3 bei beiden)
SANDBOX = "nebius"


# ---- Einheiten ohne Klone laden (Steckbrief kommt aus G0) ---------------------------------------------------------------------
def load_light(main: Path, unit: str) -> dict:
    """Wie `uptake_probe.load_unit`, aber ohne Repo-Klone: Paper, Claim, belegte Setup-Details und Paper-Text aus dem gespeicherten Lauf."""
    from repro_agent.loop import load_claims
    from repro_agent.setup_details import SetupDetail

    paper, claims = load_claims(unit)
    claim = next(c for c in claims if c.id == UNITS[unit].claim_id)
    recs = [json.loads(line) for line in (main / "runs" / "benchmark_v4.jsonl").read_text().splitlines()]
    rec = next((r for r in recs if r["paper"] == unit and r["arm"].startswith("B") and r["origin"] == "paper"), None)
    run_dir = main / rec["run_dir"] if rec else sorted((main / "runs" / unit).iterdir())[-1]
    dossier = json.loads((run_dir / "dossier.json").read_text())
    paper_text = json.loads((run_dir / "paper_text.json").read_text())["text"]
    details = [SetupDetail(**{k: v for k, v in x.items() if k in SetupDetail.__dataclass_fields__}) for x in dossier["setup_details"]]
    return {"paper": paper, "claim": claim, "details": details, "paper_text": paper_text, "run_dir": str(run_dir)}


def new_spec(g0_dir: Path, main: Path, unit: str):
    """G0-Steckbrief erneut durch den deterministischen Teil (F1, F4), ohne Extraktor-Aufruf."""
    from repro_agent.spec import rebuild_spec
    old = json.loads((g0_dir / unit / "spec.json").read_text())
    data = load_light(main, unit)
    return old, rebuild_spec(old, data["details"], data["paper_text"]), data


# ---- Offline-Pruefung --------------------------------------------------------------------------------------------------------
def _items(spec_d: dict) -> dict:
    return {i["key"]: (i["value"], i.get("enforce"), i["rank"], i["source"].split("@")[0]) for i in spec_d["items"]}


def audit(g0_dir: Path, main: Path) -> tuple[str, dict]:
    """Welche Einheiten beruehrt der deterministische Teil von F1/F4? Vergleich der G0-Steckbriefe mit den neu gerechneten, Text des Prompts
    (`format_code_grounded`), Zeitlimits aus dem G0-Log (F2/F3). Kein LLM, kein Sandbox-Aufruf."""
    from repro_agent.spec import as_dict, format_code_grounded
    rows, out = [], {}
    for unit in UNITS:
        old, spec, data = new_spec(g0_dir, main, unit)
        new_d = json.loads(json.dumps(as_dict(spec), default=str))
        a, b = _items(old), _items(new_d)
        changed = {k: {"g0": a.get(k), "g1": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        miss = {"g0": old["missing"], "g1": new_d["missing"]}
        text_note = bool(old["missing"])   # F4 (b): der Hinweistext fuer fehlende Schluessel aendert sich in jedem cg-Prompt mit missing-Liste
        touched = bool(changed) or old["missing"] != new_d["missing"] or text_note or any(i.tie for i in spec.items)
        out[unit] = {"items_changed": changed, "missing": miss, "missing_text_changed": text_note, "ties": [i.key for i in spec.items if i.tie],
                     "touched": touched, "prompt_chars": len(format_code_grounded(spec, data["details"]))}
        rows.append(f"| {unit} | {len(changed)} | {', '.join(changed) or '-'} | {', '.join(old['missing']) or '-'} -> {', '.join(new_d['missing']) or '-'} | "
                    f"{'ja' if text_note else 'nein'} | {', '.join(out[unit]['ties']) or '-'} | **{'ja' if touched else 'nein'}** |")
    head = ("| Einheit | geaenderte Eintraege | welche | missing G0 -> G1 | Hinweistext F4 (b) | ungeloeste Konflikte | beruehrt |\n"
            "|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows), out


def timeout_audit(g0_dir: Path) -> dict:
    """F2: Zeitueberschreitungen im G0-Log je Claim (Zeilen `timeout`); F3: Zeilen mit Status `fehler`, deren Dauer < 5 s war, sind ohne Text nicht erkennbar."""
    counts: dict[str, int] = {}
    log = g0_dir / "log.txt"
    for line in log.read_text().splitlines() if log.exists() else []:
        if " timeout " in line and line.startswith("["):
            counts[line[1:line.index("]")]] = counts.get(line[1:line.index("]")], 0) + 1
    return counts


# ---- Entscheidungsregel G1 ---------------------------------------------------------------------------------------------------
def load_samples(d: Path) -> list[Sample]:
    return [Sample(**json.loads(f.read_text())) for f in sorted(d.glob("*/final_*.json"))]


def decide_g1(g1: list[Sample], g0s: list[Sample]) -> dict:
    """Aggregat: Entwicklungseinheiten ausser den zwei GAT-Einheiten aus G0, die zwei GAT-Einheiten aus G1. Regel (a) bis (d) wie G0 (gleiche
    Konstanten), dazu (b') Nichtverschlechterung der in G1 erneut gemessenen Einheiten Shchur-GCN und RLinear (cg >= 2 von 3 je Einheit).
    Nicht erneut gemessene Einheiten (Kipf, NLinear, SGC, DLinear, DGI) behalten ihre G0-Zahlen."""
    gat = set(G1_GAT)
    others = [u for u in DEV_UNITS if u not in gat]
    fh = g0.hits(g0s, "heute", others) + g0.hits(g1, "heute", G1_GAT)
    fc = g0.hits(g0s, "cg", others) + g0.hits(g1, "cg", G1_GAT)
    kipf_c, kipf_h = g0.hits(g0s, "cg", ["kipf_gcn_2017"]), g0.hits(g0s, "heute", ["kipf_gcn_2017"])
    sgcn_c, sgcn_h = g0.hits(g1, "cg", ["shchur_pitfalls_2018"]), g0.hits(g0s, "heute", ["shchur_pitfalls_2018"])
    rc = g0.hits(g1, "cg", ["rlinear_2023"])
    remeasured = {u for u, _ in G1_CELLS}
    missing = sum(1 for s in g1 if s.arm == "cg" and s.code_missing) + sum(1 for s in g0s if s.arm == "cg" and s.code_missing and s.unit not in remeasured)
    complete = all(any(s.unit == u and s.arm == a for s in g1) and sum(1 for s in g1 if s.unit == u and s.arm == a) == len(TEMPS)
                   and all(s.urteil is not None for s in g1 if s.unit == u and s.arm == a) for u, a in G1_CELLS)
    a = fc >= g0.RULE_A_MIN_HITS and fc >= fh + g0.RULE_A_MIN_GAIN
    b = (kipf_c + sgcn_c) >= (kipf_h + sgcn_h) - g0.RULE_B_MAX_LOSS
    b2 = sgcn_c >= NONREG_MIN_HITS
    c = rc >= g0.RULE_C_MIN_HITS
    d = missing <= g0.RULE_D_MAX_MISSING
    return {"complete": complete, "a_hits_dev_heute": fh, "a_hits_dev_cg": fc, "a_gat_heute": g0.hits(g1, "heute", G1_GAT), "a_gat_cg": g0.hits(g1, "cg", G1_GAT),
            "a_pass": a, "b_nonreg_cg": kipf_c + sgcn_c, "b_nonreg_heute": kipf_h + sgcn_h, "b_pass": b, "b2_shchur_gcn_cg": sgcn_c, "b2_pass": b2,
            "c_hits_rlinear_cg": rc, "c_pass": c, "d_cg_code_calls_missing": missing, "d_pass": d,
            "decision": "unvollstaendig" if not complete else ("bestanden" if a and b and b2 and c and d else "nicht bestanden")}


def report(g1: list[Sample], g0s: list[Sample], spent: float | None) -> str:
    cell = lambda u, a: f"{g0.hits(g1, a, [u])}/{sum(1 for s in g1 if s.unit == u and s.arm == a)}"
    lines = ["| Einheit | Arm | Treffer G1 | Treffer G0 |", "|---|---|---|---|"]
    for u, a in G1_CELLS:
        lines.append(f"| {u} | {a} | {cell(u, a)} | {g0.hits(g0s, a, [u])}/{sum(1 for s in g0s if s.unit == u and s.arm == a)} |")
    lines += ["", g0.sample_table(g1), "", "```json", json.dumps(decide_g1(g1, g0s), indent=1, ensure_ascii=False), "```"]
    if spent is not None:
        lines.append(f"Kosten G1 (brutto, spend.jsonl): {spent:.4f} $")
    return "\n".join(lines)


# ---- Lauf ---------------------------------------------------------------------------------------------------------------------
def run_g1(g0_dir: Path, main: Path, out: Path, timeouts: dict[str, int]) -> list[Sample]:
    from repro_agent.llm import LLMClient
    from repro_agent.spec import as_dict
    from repro_agent.tasks import TASKS

    out.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    spend = Spend(out / "spend.jsonl", llm, COST_CAP_GROSS_USD)
    ask = g0.make_ask(llm, spend)
    sandbox, snapshots, snap_lock = g0.make_sandbox(SANDBOX), {}, threading.Lock()
    samples: list[Sample] = []
    samples_lock = threading.Lock()
    manifest = {}

    def unit_job(unit: str) -> None:
        old, spec, data = new_spec(g0_dir, main, unit)
        task = TASKS[data["paper"]["dataset"]]()
        claim = data["claim"]
        ud = out / unit
        ud.mkdir(exist_ok=True)
        (ud / "spec.json").write_text(json.dumps(as_dict(spec) | {"excerpt_chars": len(spec.excerpt)}, indent=1, default=str))
        snap = g0.snapshot_for(sandbox, snapshots, snap_lock, data["paper"]["dataset"], task)
        n_seeds = data["paper"].get("seeds", 5)
        arms = [a for u, a in G1_CELLS if u == unit]

        def unit_arm(arm: str) -> None:
            mk = lambda attempt_log=None: g0.make_orchestrator(sandbox, ask=ask, llm=llm, details=data["details"], cg=arm == "cg", spec=spec,
                                                                timeout_s=timeouts.get(unit, 600), attempt_log=attempt_log)
            plan_file = ud / f"plan_{arm}.md"
            if not plan_file.exists():
                if arm == "heute":   # Plan und erste Codes aus G0 byte-identisch wiederverwendet (F1/F4 betreffen den Arm heute nicht)
                    shutil.copyfile(g0_dir / unit / "plan_heute.md", plan_file)
                else:
                    plan_file.write_text(mk().plan(data["paper"], claim, task))
            plan = plan_file.read_text()

            def one(t: float) -> None:
                final = ud / f"final_{arm}_{t}.json"
                if final.exists():
                    with samples_lock:
                        samples.append(Sample(**json.loads(final.read_text())))
                    return
                s = Sample(unit, arm, t, n_seeds=n_seeds)
                orch = mk(ud / f"attempts_{arm}_{t}.jsonl")   # F8
                code_file, none_file = ud / f"code_{arm}_{t}.py", ud / f"code_{arm}_{t}.none"
                t0 = time.time()
                if not code_file.exists() and not none_file.exists():
                    if arm == "heute":
                        shutil.copyfile(g0_dir / unit / f"code_heute_{t}.py", code_file)
                    else:
                        try:
                            code_file.write_text(orch.write_code(plan, task, temperature=t))
                        except (ValueError, RuntimeError) as exc:
                            none_file.write_text(str(exc)[:300])
                if not code_file.exists():
                    s.code_missing, s.urteil, s.note = True, "nicht reproduzierbar (Setup)", "kein Code vom Modell: " + none_file.read_text()
                else:
                    g0.measure_sample(orch, s, code_file.read_text(), data, task, claim, snap, plan, ud / f"run_{arm}_{t}", n_seeds)
                s.seconds = round(time.time() - t0, 1)
                final.write_text(json.dumps(asdict(s), default=str))
                with samples_lock:
                    samples.append(s)
                print(f"[{unit}] {arm} T={t}: {s.urteil} mean={s.mean} n_ok={s.n_ok}/{s.n_seeds} hit={hit(s)} ({s.seconds:.0f}s)", flush=True)

            with ThreadPoolExecutor(max_workers=len(TEMPS)) as pool:
                list(pool.map(one, TEMPS))

        with ThreadPoolExecutor(max_workers=len(arms)) as pool:
            list(pool.map(unit_arm, arms))

    try:
        units = list(dict.fromkeys(u for u, _ in G1_CELLS))
        with ThreadPoolExecutor(max_workers=len(units)) as pool:
            for f in [pool.submit(unit_job, u) for u in units]:
                f.result()
    finally:
        sandbox.cleanup()
        g0.store(samples, out)
        write_reuse_manifest(g0_dir, out)
    return samples


def write_reuse_manifest(g0_dir: Path, out: Path) -> None:
    """Belegt, dass Plan und erste Codes des Arms heute byte-identisch aus G0 stammen (SHA-256 beider Seiten)."""
    import hashlib
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rows = {}
    for unit in G1_GAT:
        for name in [f"plan_heute.md"] + [f"code_heute_{t}.py" for t in TEMPS]:
            a, b = g0_dir / unit / name, out / unit / name
            if a.exists() and b.exists():
                rows[f"{unit}/{name}"] = {"g0": sha(a), "g1": sha(b), "identical": sha(a) == sha(b)}
    (out / "reuse_manifest.json").write_text(json.dumps(rows, indent=1))


def main() -> int:
    argv = sys.argv[1:]
    arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    g0_dir = Path(arg("--g0", "benchmark/experiments/results/recall_cg_gate_samples"))
    if "--audit" in argv:
        table, data = audit(g0_dir, Path(arg("--main")))
        text = table + "\n\nZeitueberschreitungen je Claim im G0-Log (F2): " + json.dumps(timeout_audit(g0_dir)) + "\n"
        Path(arg("--out")).write_text(text)
        Path(arg("--out")).with_suffix(".json").write_text(json.dumps(data, indent=1, default=str))
        print(text)
        return 0
    if "--score" in argv:
        out = Path(arg("--score"))
        spent = Spend(out / "spend.jsonl", type("L", (), {"gross": staticmethod(lambda x: x)})(), 1e9).total() if (out / "spend.jsonl").exists() else None
        print(report(load_samples(out), load_samples(g0_dir), spent))
        return 0
    if "--run" in argv:
        timeouts = json.loads(Path(arg("--timeouts")).read_text()) if arg("--timeouts") else {}
        try:
            samples = run_g1(g0_dir, Path(arg("--main")), Path(arg("--out")), timeouts)
        except CostStop as exc:
            print(f"STOPP: {exc}. Wiederaufnahme mit demselben Befehl (out/ bleibt).", file=sys.stderr, flush=True)
            return 2
        print(report(samples, load_samples(g0_dir), None))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())

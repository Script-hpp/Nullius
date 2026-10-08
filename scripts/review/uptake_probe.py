"""V1 (Umsetzungstest) des Steckbrief-Plans: kommen die entscheidenden Fakten im Code an? Vorab festgelegt in
benchmark/experiments/recall_v1_uptake.md (dort stehen Einheiten, Fakten, Arme und die Entscheidungsregel).

Dieses Skript wird VOR dem ersten Probe-Aufruf committet. Fakt-Checks (AST/Regex) und `decide` aendern sich danach nicht mehr.

  uv run python scripts/review/uptake_probe.py --main <Checkout mit runs/> --clones <Verzeichnis mit Repo-Klonen> --out <Ausgabe>
  uv run python scripts/review/uptake_probe.py --score <Ausgabe>        (nur auswerten, kein LLM)

Modell-Aufrufe: nur ueber LLMClient (Provider per Umgebung, Prototyp: LLM_PROVIDER=openrouter, 0 $). Keine Tavily-, keine
Sandbox-Aufrufe: Paper-Text und Setup-Details kommen aus gespeicherten Dossiers, Repo-Dateien aus lokalen Klonen (nur Lesen).
"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TEMPS = (0.2, 0.5, 0.8)
ARMS = ("heute", "steckbrief", "steckbrief_pruefung")
THRESHOLD_ABS, THRESHOLD_GAIN, MAX_FAILED_CALLS = 0.7, 0.25, 2
COST_CAP_GROSS_USD, COST_CAP_MARGIN_USD = 1.20, 0.03   # Addendum 2026-10-08: harte Grenze brutto fuer ganz V1


class CostCapReached(BaseException):   # nicht Exception: loop.py faengt RuntimeError/Exception
    """Kostengrenze erreicht: Lauf abbrechen, Teilergebnis berichten."""


# ---- Fakt-Checks (deterministisch, ohne LLM) --------------------------------------------------------------------------------
class Code:
    """Agenten-Code ohne Kommentare und Docstrings (ast.unparse): ein Plan-Kommentar zaehlt nie als umgesetzt."""

    def __init__(self, src: str):
        self.ok = True
        try:
            self.tree = ast.parse(src)
            self.text = ast.unparse(self.tree)
        except (SyntaxError, ValueError):
            self.ok, self.tree, self.text = False, ast.parse(""), ""
        self.consts = {n.value for n in ast.walk(self.tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                       and not isinstance(n.value, bool)}

    def const(self, x: float) -> bool:
        return any(abs(float(c) - x) < 1e-12 for c in self.consts)

    def has(self, pattern: str) -> bool:
        return bool(re.search(pattern, self.text, re.I | re.S))

    def subtract_last(self) -> bool:
        """Letzten Wert des Fensters (x[:, -1:, :] oder x[:, -1, :]) abziehen und am Ende wieder addieren."""
        names = set()
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if any(isinstance(s, ast.Subscript) and re.search(r"-1\b", ast.unparse(s.slice)) for s in ast.walk(n.value)):
                    names.add(n.targets[0].id)
        for t in names:
            sub = add = False
            for n in ast.walk(self.tree):
                parts = []
                if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Sub, ast.Add)):
                    parts = [(type(n.op), n.right)]
                elif isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Sub, ast.Add)):
                    parts = [(type(n.op), n.value)]
                for op, side in parts:
                    if isinstance(side, ast.Name) and side.id == t or (isinstance(side, ast.Call) and t in ast.unparse(side)):
                        sub |= op is ast.Sub
                        add |= op is ast.Add
            if sub and add:
                return True
        return False


ROWNORM = (r"row_?normali[sz]e|row_?sum|\.sum\(\s*(axis\s*=\s*|dim\s*=\s*)?1\b|norm\s*=\s*['\"]l1['\"]|preprocess_features"
           r"|\bnormalize\(")
COLSTD = r"StandardScaler|\.std\(|np\.std|torch\.std|\bscale\(|\.mean\(\s*(axis\s*=\s*|dim\s*=\s*)?0\b"
HALVE = r"0\.5\s*\*\*|\*\s*0\.5\b|\*=\s*0\.5\b|gamma\s*=\s*0\.5\b|\b0\.5\s*\*\s*(lr|self\.lr|learning)"
DECOMP = r"series_decomp|moving_avg|decomposition|\btrend\b|seasonal"

# id -> (Klasse, Check). Klassen: D geliefert aber nicht befolgt, E im Repo nicht herausgezogen, R Datei nie gelesen,
# R* falsche Datei uebernommen, N neues Paper (Regel 2).
FACTS = {
    "zeng_dlinear_2023": {
        "seq336": ("D", lambda c: c.const(336)),
        "lr0.005": ("D", lambda c: c.const(0.005)),
        "halve_each_epoch": ("D", lambda c: c.has(HALVE)),
    },
    "zeng_nlinear_2023": {
        "seq336": ("D", lambda c: c.const(336)),
        "lr0.005": ("D", lambda c: c.const(0.005)),
        "halve_each_epoch": ("D", lambda c: c.has(HALVE)),
        "subtract_last": ("R*", lambda c: c.subtract_last()),
        "no_decomposition": ("R*", lambda c: not c.has(DECOMP)),
    },
    "wu_sgc_2019": {
        "feat_row_l1": ("E", lambda c: c.has(ROWNORM)),
        "no_col_standardize": ("E", lambda c: not c.has(COLSTD)),
    },
    "velickovic_gat_2018": {
        "feat_row_l1": ("E", lambda c: c.has(ROWNORM)),
    },
    "velickovic_dgi_2019": {
        "feat_row_l1": ("E", lambda c: c.has(ROWNORM)),
        "graph_conv": ("E", lambda c: c.has(r"adj\w*\s*@|@\s*\w*adj|torch\.(sp)?mm\(|spmm|matmul\(|torch\.bmm|sparse\.mm")),
        "prelu": ("E", lambda c: c.has(r"PReLU|F\.prelu|nn\.prelu")),
        "hid512": ("E", lambda c: c.const(512)),
    },
    "shchur_gat_2018": {
        "additive_attention": ("R", lambda c: c.has(r"leaky_?relu") and c.has(r"softmax")),
        "self_loops": ("R", lambda c: c.has(r"\beye\(|add_self_loops|fill_diagonal|setdiag|self_loops?")),
        "both_directions": ("R", lambda c: c.has(r"\.T\b|\.t\(\)|transpose|maximum\(|to_undirected|symmetri|adj\w*\s*\+\s*adj\w*")),
    },
    "rlinear_2023": {
        "seq336": ("N", lambda c: c.const(336)),
        "lr0.005": ("N", lambda c: c.const(0.005)),
        "halve_each_epoch": ("N", lambda c: c.has(HALVE)),
        "batch128": ("N", lambda c: c.const(128)),
        "revin": ("N", lambda c: c.has(r"\.mean\(") and c.has(r"std|var\(") and c.has(r"/\s*\(?\w*(std|stdev|sigma)\w*")
                  and c.has(r"\*\s*\(?\w*(std|stdev|sigma)\w*") and c.has(r"\+\s*\(?\w*(mean|mu|means|avg)\w*")),
    },
}
DEV_UNITS = [u for u in FACTS if u != "rlinear_2023"]
NEW_UNITS = ["rlinear_2023"]
# Gegenpruefung auf der Steckbrief-Seite (Zuordnung S1 vs. S2/S3): steht der Fakt (Wert) im Steckbrief-Text?
SPEC_HAS = {"seq336": r"\b336\b", "lr0.005": r"0\.005|5e-0?3", "halve_each_epoch": r"0\.5|halv", "batch128": r"\b128\b",
            "hid512": r"\b512\b", "prelu": r"prelu", "feat_row_l1": r"row[- ]?normali|sum to 1|l1|preprocess_features",
            "subtract_last": r"last (value|element)|seq_last|subtract", "revin": r"revin|reversible|mean and (std|standard)",
            "additive_attention": r"additive|leaky", "self_loops": r"self[- ]loop", "both_directions": r"both direction|symmetri|undirected|reverse",
            "graph_conv": r"graph conv|adjacen|propagat", "no_col_standardize": r"row|l1", "no_decomposition": r"."}


def check_facts(unit: str, src: str) -> dict[str, bool]:
    c = Code(src)
    return {fid: bool(c.ok and fn(c)) for fid, (_, fn) in FACTS[unit].items()}


# ---- Entscheidungsregel (docs/recall_steckbrief_plan.md Abschnitt 5, vorab) -----------------------------------------------------
@dataclass
class Sample:
    unit: str
    arm: str
    temp: float
    code: str | None            # None = kein Code (leer oder abgeschnitten)
    facts: dict[str, bool]
    n_override: int = 0
    override_broke: bool = False


def share(samples: list[Sample], arm: str, units: list[str]) -> float | None:
    got = [v for s in samples if s.arm == arm and s.unit in units for v in s.facts.values()]
    return sum(got) / len(got) if got else None


def decide(samples: list[Sample]) -> dict:
    """Vier Bedingungen; alle muessen gelten. Faktanteil = erfuellte Fakt-Instanzen / alle (Einheit x Probe x Fakt); ein Aufruf ohne Code
    zaehlt alle seine Fakten als nicht erfuellt."""
    a1, a3 = share(samples, "heute", DEV_UNITS), share(samples, "steckbrief_pruefung", DEV_UNITS)
    n1, n3 = share(samples, "heute", NEW_UNITS), share(samples, "steckbrief_pruefung", NEW_UNITS)
    failed = sum(1 for s in samples if s.arm == "steckbrief_pruefung" and s.code is None)
    broke = sum(1 for s in samples if s.arm == "steckbrief_pruefung" and s.override_broke)
    c1 = a3 is not None and a1 is not None and a3 >= THRESHOLD_ABS and a3 - a1 >= THRESHOLD_GAIN
    c2 = n3 is not None and n1 is not None and n3 >= n1
    c3 = failed <= MAX_FAILED_CALLS
    c4 = broke == 0
    return {"share_heute_dev": a1, "share_pruefung_dev": a3, "share_steckbrief_dev": share(samples, "steckbrief", DEV_UNITS),
            "share_heute_new": n1, "share_pruefung_new": n3, "failed_calls_pruefung": failed, "override_broke": broke,
            "c1_threshold_and_gain": c1, "c2_new_not_below": c2, "c3_failed_calls": c3, "c4_override_safe": c4,
            "go": bool(c1 and c2 and c3 and c4)}


def override_broke(src: str) -> bool:
    """spec_override darf den Code nicht zerstoeren: wo ein Override-Kommentar steht, muss der Code parsen und CONFIG literal sein."""
    if "# spec override" not in src:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True
    cfg = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets)]
    return not cfg


# ---- Lauf --------------------------------------------------------------------------------------------------------------------
def load_unit(main: Path, clones: Path, unit: str) -> dict:
    from repro_agent.loop import load_claims
    from repro_agent.repo_source import read_repo
    from repro_agent.setup_details import Doc, SetupDetail
    from repro_agent.spec import model_name_of

    store = main / "runs" / ("benchmark_v4.jsonl")
    paper, claims = load_claims(unit)
    recs = [json.loads(line) for line in store.read_text().splitlines()]
    rec = next((r for r in recs if r["paper"] == unit and r["arm"].startswith("B") and r["origin"] == "paper"), None)
    run_dir = main / rec["run_dir"] if rec else None
    if run_dir is None:   # neue Einheit: neuester Lauf mit gespeichertem Dossier
        run_dir = sorted((main / "runs" / unit).iterdir())[-1]
    dossier = json.loads((run_dir / "dossier.json").read_text())
    paper_text = json.loads((run_dir / "paper_text.json").read_text())["text"]
    repo = dossier["repo"]
    clone = clones / repo["url"].rstrip("/").split("github.com/")[-1].replace("/", "_")
    if not clone.exists():
        clone = clones / repo["url"].rstrip("/").rsplit("/", 1)[-1].lower()
    chosen = [p for p in repo["files"] if (clone / p).exists()]   # Proxy der LLM-Dateiwahl: die Dateien des gespeicherten Laufs

    class _Sandbox:
        def base_snapshot(self, *a, **k): return "s"
        def branch(self, s): return "b"
        def get_file(self, b, path):
            rel = path.removeprefix("/repo/")
            if rel == ".head_commit":
                return repo["commit"].encode()
            return (clone / rel).read_bytes()
        def run(self, b, cmd, t):
            import subprocess
            from types import SimpleNamespace
            return SimpleNamespace(stdout=subprocess.run(["git", "-C", str(clone), "ls-files"], capture_output=True, text=True).stdout,
                                   stderr="", returncode=0)

    llm_stub = type("L", (), {"complete": lambda self, *a, **k: type("R", (), {"content": json.dumps(chosen)})()})()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dump = read_repo(_Sandbox(), llm_stub, repo["url"], claims[0].text, paper["dataset"], focus_model=True, unit_key=unit,
                         model=model_name_of(claims[0].text), fixture_dir=Path(tmp))
    docs = [Doc("paper", paper_text)]
    readme = (clone / "README.md")
    if readme.exists():
        docs.append(Doc("repo-readme", readme.read_text(errors="replace")))
    docs += [Doc(f"repo:{p}@{dump.commit[:7]}", t) for p, t in dump.files.items()]
    details = [SetupDetail(**{k: v for k, v in x.items() if k in SetupDetail.__dataclass_fields__}) for x in dossier["setup_details"]]
    return {"paper": paper, "claim": claims[0], "docs": docs, "details": details}


def make_orchestrator(ask, unit_data: dict, arm: str, spec, events: list):
    from repro_agent.loop import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair, o.setup_v2 = unit_data["details"], True, True   # Produktkonfiguration wie Benchmark v4
    o.spec_mode, o.spec = arm != "heute", spec if arm != "heute" else None
    o._ask = ask
    o._emit = lambda type, **kw: events.append((type, kw))
    o._announce_code = lambda *a, **k: None
    o._warn = lambda text: events.append(("warning", {"text": text}))
    if arm == "steckbrief":   # nur S1 im Prompt: weder CONFIG-Pflicht noch Override
        o._spec_note = lambda: ""
        o._enforce_spec = lambda code: code
    return o


def run_probe(ask_for, main: Path, clones: Path, out: Path, units: list[str], arms: tuple[str, ...] = ARMS,
              temps: tuple[float, ...] = TEMPS) -> list[Sample]:
    """ask_for(role) -> ask(role, system, user, max_tokens=None, temperature=0.2) mit .content/.finish_reason."""
    out.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    try:
        _probe_units(ask_for, main, clones, out, units, arms, temps, samples)
    except CostCapReached as exc:   # Teilergebnis speichern und berichten
        print(f"ABBRUCH (Kostengrenze): {exc}; {len(samples)} Proben gespeichert", file=sys.stderr, flush=True)
        (out / "ABORTED_COST_CAP.txt").write_text(str(exc))
    store(samples, out)
    return samples


def _probe_units(ask_for, main, clones, out, units, arms, temps, samples) -> None:
    from repro_agent.spec import build_spec, format_spec
    from repro_agent.tasks import TASKS

    for unit in units:
        data = load_unit(main, clones, unit)
        task = TASKS[data["paper"]["dataset"]]()
        ask = ask_for
        spec = build_spec(lambda system, user: ask("extract", system, user, max_tokens=12000, temperature=0.0).content,
                          data["details"], data["docs"], data["claim"].text, data["paper"]["dataset"])
        (out / f"{unit}_spec.json").write_text(json.dumps({"formatted": format_spec(spec), "dropped": spec.dropped,
                                                           "missing": spec.missing}, indent=1, default=str))
        for arm in arms:
            events: list = []
            o = make_orchestrator(ask, data, arm, spec, events)
            plan = o.plan(data["paper"], data["claim"], task)
            for t in temps:
                code = None
                try:
                    code = o.write_code(plan, task, temperature=t)
                    if arm == "steckbrief_pruefung":
                        code, _, _ = o._spec_conform(data["claim"], None, plan, task, code)
                except (ValueError, RuntimeError) as exc:   # kein Code nach drei Antworten: leer oder abgeschnitten
                    events.append(("code_missing", {"reason": str(exc)[:200]}))
                s = Sample(unit, arm, t, code, check_facts(unit, code) if code else {f: False for f in FACTS[unit]},
                           n_override=sum(1 for ty, _ in events if ty == "spec_override"),
                           override_broke=bool(code) and override_broke(code))
                samples.append(s)
                (out / f"{unit}__{arm}__{t}.py").write_text(code or "# kein Code\n")
        (out / f"{unit}_events.json").write_text(json.dumps(events[-50:], default=str)[:20000])


def store(samples: list[Sample], out: Path) -> None:
    with open(out / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__) + "\n")
    # verblindete Kopien fuer eine Person, die den Arm nicht kennt (Schluessel getrennt)
    blind = out / "blind"
    blind.mkdir(exist_ok=True)
    ids = list(range(len(samples)))
    random.Random(0).shuffle(ids)
    key = {}
    for i, s in zip(ids, samples):
        (blind / f"{i:03d}.py").write_text(s.code or "# kein Code\n")
        key[f"{i:03d}"] = [s.unit, s.arm, s.temp]
    (out / "blind_key.json").write_text(json.dumps(key))


def report(samples: list[Sample]) -> str:
    d = decide(samples)
    arms = [a for a in ARMS if any(s.arm == a for s in samples)]
    lines = ["| Einheit | Fakt | Klasse | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    for unit, facts in FACTS.items():
        for fid, (cls, _) in facts.items():
            cells = []
            for arm in arms:
                got = [s.facts.get(fid, False) for s in samples if s.unit == unit and s.arm == arm]
                cells.append(f"{sum(got)}/{len(got)}" if got else "-")
            lines.append(f"| {unit} | {fid} | {cls} | " + " | ".join(cells) + " |")
    lines += ["", "```json", json.dumps(d, indent=1), "```"]
    return "\n".join(lines)


def main() -> int:
    argv = sys.argv[1:]
    if "--score" in argv:
        out = Path(argv[argv.index("--score") + 1])
        samples = [Sample(**json.loads(line)) for line in (out / "samples.jsonl").read_text().splitlines()]
        print(report(samples))
        return 0
    arg = lambda n: argv[argv.index(n) + 1]
    from repro_agent.llm import LLMClient
    llm = LLMClient()

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        spent = llm.gross(llm.total_cost_usd)
        if spent >= COST_CAP_GROSS_USD - COST_CAP_MARGIN_USD:
            raise CostCapReached(f"{spent:.3f} $ brutto >= Grenze {COST_CAP_GROSS_USD} $ minus Marge {COST_CAP_MARGIN_USD} $")
        return llm.complete(role, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            temperature=temperature, max_tokens=max_tokens)

    units = arg("--units").split(",") if "--units" in argv else DEV_UNITS + NEW_UNITS
    arms = tuple(arg("--arms").split(",")) if "--arms" in argv else ARMS
    temps = tuple(float(x) for x in arg("--temps").split(",")) if "--temps" in argv else TEMPS
    samples = run_probe(ask, Path(arg("--main")), Path(arg("--clones")), Path(arg("--out")), units, arms, temps)
    print(report(samples))
    print(f"Kosten dieses Laufs: {llm.total_cost_usd:.4f} $ netto, {llm.gross(llm.total_cost_usd):.4f} $ brutto")
    return 0


if __name__ == "__main__":
    sys.exit(main())

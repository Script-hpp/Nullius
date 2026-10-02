"""Schritt 2 des Vorversuchs "Nachlauf": dieselben Laeufe in der Nebius-Sandbox (Handarbeit, kein Agent, kein LLM).

NICHT automatisch ausgefuehrt; der Reviewer startet es:
  python scripts/prestudy_nachlauf/sandbox_run.py --unit rlinear            # Seeds: Repo-Seed, 0, 1, 2, 3
  python scripts/prestudy_nachlauf/sandbox_run.py --unit dlinear --seeds repo,0
  python scripts/prestudy_nachlauf/sandbox_run.py --unit rlinear --dry-run   # nur Befehle drucken, kein Sandbox-Client
  --snapshot <tag>                                                           # vorhandenen Basis-Snapshot wiederverwenden

Ablauf: Basis-Snapshot (Label prestudy-nachlauf: numpy<2 und fehlende Pakete, beide Repos auf festen Commits, ETTh1.csv mit SHA-256-Pruefung),
dann je Einheit und Seed EIN Branch, streng nacheinander:
  Seed-Kopie (nur Seeds != Repo-Seed) pruefen -> Training unter `unshare -n` (600 s) -> Adapter unter `unshare -n` ->
  git status/diff im Klon (muss leer sein) -> erst dann Harness frisch nach /opt/harness-<zufall> und etth1_eval.py unveraendert (mit Netz).
Pro Seed eine JSON-Zeile (stdout und Ergebnisdatei), Rohausgabe nach benchmark/experiments/results/nachlauf_prestudy_sandbox_<einheit>.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent.tasks import HARNESS_DIR, SANDBOX_HARNESS_DIR, etth1_task  # noqa: E402  (Harness-Dateien und Image wie im Loop)

SNAPSHOT_LABEL = "prestudy-nachlauf"
TIMEOUT_S = 600
ETTH1_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
ADAPTER_DIR = "/opt/adapter"                 # common.py, Adapter; etth1_data.py nur fuer die Zuordnungspruefung des Adapters
ADAPTER_HARNESS_DIR = f"{ADAPTER_DIR}/harness"
SEED_DIR = "/opt/seed"
WORK = "/work"
OUTPUT = "/output"

REPOS = {
    "rtsf": ("https://github.com/plumprc/RTSF.git", "0fec00104f754f4fbf795b9b4da5fa2459b32e76"),
    "ltsf": ("https://github.com/cure-lab/LTSF-Linear.git", "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"),
}
UNITS = {
    "rlinear": dict(
        repo="rtsf", repo_seed=1024, script="run.py", seed_copy="rtsf_run_seed{n}.py", adapter="adapter_rtsf.py", droplast="128",
        # script.md:3 mit `--model RLinear` statt `Linear` (siehe Ergebnis Schritt 1)
        args="--is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 "
             "--learning_rate 0.005 --model RLinear --rev --seq_len 336 --pred_len 96 --gpu 3",
    ),
    "dlinear": dict(
        repo="ltsf", repo_seed=2021, script="run_longExp.py", seed_copy="ltsf_run_seed{n}.py", adapter="adapter_ltsf.py", droplast="32",
        # Zeile Horizont 96 aus scripts/EXP-LongForecasting/Linear/etth1.sh
        args="--is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear --data ETTh1 "
             "--features M --seq_len 336 --pred_len 96 --enc_in 7 --des Exp --itr 1 --batch_size 32 --learning_rate 0.005",
    ),
}


# ---- Befehle (rein, ohne Sandbox) -------------------------------------------------------------------------------------

def setup_cmds() -> list[str]:
    from repro_agent.harness import etth1_data  # nur fuer die URL
    clone = []
    for name, (url, commit) in REPOS.items():
        clone.append(
            f"git clone -q {url} {WORK}/{name} && git -C {WORK}/{name} checkout -q {commit} && "
            f"test \"$(git -C {WORK}/{name} rev-parse HEAD)\" = {commit}"
        )
    fetch = (
        f"mkdir -p /opt/data && python -c \"import urllib.request; urllib.request.urlretrieve('{etth1_data.URL}', '/opt/data/ETTh1.csv')\" && "
        f"echo '{ETTH1_SHA256}  /opt/data/ETTh1.csv' | sha256sum -c - && "
        + " && ".join(f"mkdir -p {WORK}/{n}/dataset && cp /opt/data/ETTh1.csv {WORK}/{n}/dataset/ETTh1.csv" for n in REPOS)
    )
    return [
        "apt-get update -qq && apt-get install -y -qq git ca-certificates > /dev/null",
        # numpy<2 fuer beide Repos (RTSF nutzt np.Inf); Pakete, die das Image nicht hat
        "pip install -q 'numpy<2' pandas scikit-learn matplotlib torch --extra-index-url https://download.pytorch.org/whl/cpu",
        *clone,
        fetch,
    ]


PROBE_CMD = (
    "python --version; pip freeze; echo '__DU__'; du -sxh / 2>/dev/null; du -sh /work /usr/local /opt 2>/dev/null; "
    f"sha256sum {WORK}/rtsf/dataset/ETTh1.csv {WORK}/ltsf/dataset/ETTh1.csv; "
    f"git -C {WORK}/rtsf rev-parse HEAD; git -C {WORK}/ltsf rev-parse HEAD"
)


def timed(cmd: str, marker: str) -> str:
    """Wanduhr im Branch (ms), Exit-Code des Befehls bleibt erhalten."""
    return f"S=$(date +%s%N); {cmd}; RC=$?; E=$(date +%s%N); echo \"{marker} $(( (E-S)/1000000 ))\"; exit $RC"


def seed_script(unit: dict, seed: int) -> str:
    if seed == unit["repo_seed"]:
        return unit["script"]                      # unveraendertes Startskript
    return f"{SEED_DIR}/{unit['seed_copy'].format(n=seed)}"


def seed_check_cmd(unit: dict, seed: int) -> str:
    """Seed-Kopie darf sich nur in der Zeile fix_seed vom Original unterscheiden: genau zwei Diff-Zeilen (eine alt, eine neu)."""
    clone, script = f"{WORK}/{unit['repo']}", unit["script"]
    copy = seed_script(unit, seed)
    return f"diff {clone}/{script} {copy}; echo \"__DIFFLINES__ $(diff {clone}/{script} {copy} | grep -c '^[<>]')\"; grep -c '^fix_seed = {seed}$' {copy}"


def train_cmd(unit: dict, seed: int) -> str:
    clone = f"{WORK}/{unit['repo']}"
    py = f"PYTHONPATH={clone} unshare -n python -u {seed_script(unit, seed)} {unit['args']}"
    return f"mkdir -p {OUTPUT} && cd {clone} && " + timed(py, "__WALL_MS__")


def adapter_cmd(unit: dict) -> str:
    clone = f"{WORK}/{unit['repo']}"
    inner = (
        f"cd {clone} && CK=$(ls {clone}/checkpoints/*/checkpoint.pth) && "
        f"PRESTUDY_HARNESS_DIR={ADAPTER_HARNESS_DIR} python {ADAPTER_DIR}/{unit['adapter']} --repo {clone} --ckpt \"$CK\" "
        f"--csv {clone}/dataset/ETTh1.csv --out {OUTPUT}"
    )
    return timed(f"unshare -n sh -c {shlex.quote(inner)}", "__WALL_MS__")


def git_cmd(unit: dict) -> str:
    clone = f"{WORK}/{unit['repo']}"
    return f"cd {clone} && echo \"__GIT_STATUS__[$(git status --short)]\" && echo \"__GIT_DIFFSTAT__[$(git diff --stat)]\""


MAPPING_CMD = f"cat {OUTPUT}/mapping.json"


def eval_cmd(harness_dir: str) -> str:
    return f"python {harness_dir}/etth1_eval.py"   # unveraendert, mit Netz (laedt ETTh1 selbst), wie im Loop


def adapter_files() -> dict[str, bytes]:
    files = {f"{ADAPTER_DIR}/common.py": (HERE / "common.py").read_bytes(),
             f"{ADAPTER_HARNESS_DIR}/etth1_data.py": (HARNESS_DIR / "etth1_data.py").read_bytes()}
    for name in ("adapter_rtsf.py", "adapter_ltsf.py"):
        files[f"{ADAPTER_DIR}/{name}"] = (HERE / name).read_bytes()
    return files


def seed_files(unit: dict, seed: int) -> dict[str, bytes]:
    if seed == unit["repo_seed"]:
        return {}
    name = unit["seed_copy"].format(n=seed)
    return {f"{SEED_DIR}/{name}": (HERE / "seed_copies" / name).read_bytes()}


def harness_files(harness_dir: str) -> dict[str, bytes]:
    return {f"{harness_dir}/{n}": (HARNESS_DIR / n).read_bytes() for n in etth1_task().harness_files}


def parse_seeds(text: str, unit: dict) -> list[int]:
    return [unit["repo_seed"] if t.strip() == "repo" else int(t) for t in text.split(",") if t.strip()]


def _marker(text: str, name: str) -> str | None:
    m = re.search(rf"^{name} (.*)$", text, re.M)
    return m.group(1).strip() if m else None


def _bracket(text: str, name: str) -> str | None:
    m = re.search(rf"^{name}\[(.*?)\]\s*$", text, re.M | re.S)
    return m.group(1) if m else None


# ---- Ablauf ---------------------------------------------------------------------------------------------------------

class Report:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.write_text(f"# Rohausgaben Vorversuch Nachlauf, Schritt 2 (Nebius-Sandbox), erzeugt von sandbox_run.py\n")

    def raw(self, title: str, text: str) -> None:
        block = f"\n########## {title}\n{text}\n"
        if self.path:
            with open(self.path, "a") as f:
                f.write(block)

    def line(self, rec: dict) -> None:
        s = json.dumps(rec, ensure_ascii=False)
        print(s, flush=True)
        self.raw("JSON", s)


def _filter(text: str) -> str:
    return "\n".join(l[:400] for l in text.splitlines() if "iters:" not in l and "speed:" not in l)


def run_seed(sb, snapshot: str, name: str, unit: dict, seed: int, report: Report) -> dict:
    rec: dict = {"unit": name, "seed": seed, "repo_seed": seed == unit["repo_seed"], "status": "ok"}
    branch = sb.branch(snapshot)
    sb.put_files(branch, {**adapter_files(), **seed_files(unit, seed)})

    if not rec["repo_seed"]:
        c = sb.run(branch, seed_check_cmd(unit, seed), 60)
        report.raw(f"{name} seed {seed}: Seed-Kopie gegen Original", c.stdout + c.stderr)
        if _marker(c.stdout, "__DIFFLINES__") != "2":
            return {**rec, "status": "seed_copy_mismatch"}

    t0 = time.monotonic()
    tr = sb.run(branch, train_cmd(unit, seed), TIMEOUT_S)
    rec["train_wall_s_client"] = round(time.monotonic() - t0, 1)
    ms = _marker(tr.stdout, "__WALL_MS__")
    rec["train_wall_s"] = round(int(ms) / 1000, 1) if ms else None
    rec["train_exit"] = tr.exit_code
    report.raw(f"{name} seed {seed}: Training (unshare -n), Exit {tr.exit_code}, timed_out={tr.timed_out}", _filter(tr.stdout) + "\n--- stderr\n" + tr.stderr[-3000:])
    own = re.findall(r"^mse:\s*([0-9.eE+-]+)", tr.stdout, re.M)
    rec["repo_mse"] = float(own[-1]) if own else None
    if tr.timed_out:
        return {**rec, "status": "timeout"}
    if tr.exit_code != 0:
        return {**rec, "status": "train_failed"}

    ad = sb.run(branch, adapter_cmd(unit), TIMEOUT_S)
    ms = _marker(ad.stdout, "__WALL_MS__")
    rec["adapter_wall_s"] = round(int(ms) / 1000, 1) if ms else None
    rec["adapter_exit"] = ad.exit_code
    report.raw(f"{name} seed {seed}: Adapter (unshare -n), Exit {ad.exit_code}", ad.stdout + "\n--- stderr\n" + ad.stderr[-3000:])
    mp = sb.run(branch, MAPPING_CMD, 30)
    try:
        rec["mapping"] = json.loads(mp.stdout)
    except json.JSONDecodeError:
        rec["mapping"] = None
    if ad.exit_code != 0 or not (rec["mapping"] or {}).get("ok"):
        return {**rec, "status": "mapping_failed" if rec["mapping"] else "adapter_failed"}

    g = sb.run(branch, git_cmd(unit), 60)
    rec["git_exit"] = g.exit_code
    status, diffstat = _bracket(g.stdout, "__GIT_STATUS__"), _bracket(g.stdout, "__GIT_DIFFSTAT__")
    rec["git_status"], rec["git_diffstat"] = status, diffstat
    report.raw(f"{name} seed {seed}: Klon unveraendert?", g.stdout + g.stderr)
    if g.exit_code != 0 or status != "" or diffstat != "":
        return {**rec, "status": "repo_changed"}

    # erst jetzt der Harness, frisch aus unserem Repo, in ein Verzeichnis mit Zufallsnamen
    hdir = f"{SANDBOX_HARNESS_DIR}-{uuid.uuid4().hex[:10]}"
    sb.put_files(branch, harness_files(hdir))
    e = sb.run(branch, eval_cmd(hdir), TIMEOUT_S)
    rec["harness_exit"] = e.exit_code
    report.raw(f"{name} seed {seed}: Harness, Exit {e.exit_code}", e.stdout + "\n--- stderr\n" + e.stderr[-3000:])
    try:
        out = json.loads(e.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {**rec, "status": "harness_unreadable"}
    if out.get("status") != "ok":
        return {**rec, "status": "ungueltig", "reason": out.get("reason")}
    rec["harness_mse_all"] = out["metrics"]["mse"]
    rec["metrics_droplast"] = out.get("metrics_droplast")
    rec["droplast_batch"] = unit["droplast"]
    return rec


def run_unit(sb, snapshot: str, name: str, seeds: list[int], out_dir: Path | None) -> list[dict]:
    unit = UNITS[name]
    report = Report(out_dir / f"nachlauf_prestudy_sandbox_{name}.txt" if out_dir else None)
    probe = sb.branch(snapshot)
    p = sb.run(probe, PROBE_CMD, 300)
    print(p.stdout, flush=True)
    report.raw("Snapshot: Python, pip freeze, Groesse (du), Dateihashes, Commits", p.stdout + p.stderr[-1000:])
    results = []
    for seed in seeds:                             # streng nacheinander, je Seed ein eigener Branch
        results.append(run_seed(sb, snapshot, name, unit, seed, report))
        report.line(results[-1])
    return results


def dry_run(name: str, seeds: list[int]) -> None:
    unit = UNITS[name]
    print("# Snapshot-Setup (Label %s, Image %s):" % (SNAPSHOT_LABEL, etth1_task().image))
    for c in setup_cmds():
        print("  $", c)
    print("# Probe im Snapshot:\n  $", PROBE_CMD)
    for seed in seeds:
        print(f"\n# {name} seed {seed} (neuer Branch)")
        print("  put_files:", sorted({**adapter_files(), **seed_files(unit, seed)}))
        if seed != unit["repo_seed"]:
            print("  $", seed_check_cmd(unit, seed))
        for c in (train_cmd(unit, seed), adapter_cmd(unit), MAPPING_CMD, git_cmd(unit)):
            print("  $", c)
        print(f"  put_files: {SANDBOX_HARNESS_DIR}-<zufall>/", [Path(k).name for k in harness_files("X")])
        print("  $", eval_cmd(f"{SANDBOX_HARNESS_DIR}-<zufall>"))


def main(argv: list[str] | None = None, sandbox_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=sorted(UNITS), required=True)
    ap.add_argument("--seeds", default="repo,0,1,2,3", help="Komma-Liste, `repo` = Seed des Repos")
    ap.add_argument("--dry-run", action="store_true", help="nur Befehle drucken, keinen Sandbox-Client erzeugen")
    ap.add_argument("--snapshot", help="vorhandenen Basis-Snapshot (Tag) verwenden statt neu zu bauen")
    ap.add_argument("--out-dir", default=str(ROOT / "benchmark" / "experiments" / "results"))
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds, UNITS[a.unit])
    if a.dry_run:
        dry_run(a.unit, seeds)
        return 0
    if sandbox_factory is None:
        from repro_agent.sandbox import get_sandbox
        sandbox_factory = lambda: get_sandbox("nebius")  # noqa: E731
    sb = sandbox_factory()
    snapshot = a.snapshot or sb.base_snapshot(etth1_task().image, setup_cmds(), label=SNAPSHOT_LABEL)
    print("snapshot:", snapshot, flush=True)
    results = run_unit(sb, snapshot, a.unit, seeds, Path(a.out_dir))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

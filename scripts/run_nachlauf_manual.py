"""Modus Nachlauf, Meilenstein M1: Ablauf mit Hand-Specs (Startbefehle und Adapter von Hand, kein Agent, kein LLM, kein Tavily).

Der Reviewer startet es (echte Nebius-Sandbox); die Entwicklung laeuft nur mit Fake-Sandbox (tests/test_nachlauf.py).
  uv run python scripts/run_nachlauf_manual.py --unit rlinear --dry-run
  uv run python scripts/run_nachlauf_manual.py --unit rlinear                    # Seeds: Repo-Seed, 0, 1, 2, 3; 8 P2-Versaetze
  uv run python scripts/run_nachlauf_manual.py --unit dlinear --seeds repo,0 --offsets 2
  uv run python scripts/run_nachlauf_manual.py --negative a|b|c [--variant shift|permute] [--dry-run]   # M2: Negativkontrollen, muessen abgelehnt werden
Ausgabe: runs/nachlauf_manual/<unit>/<zeitstempel>/{dossier.json, events.jsonl}, eine Zusammenfassungszeile je Seed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent import nachlauf as nl  # noqa: E402
from repro_agent.harness import independence  # noqa: E402

ADAPTERS = ROOT / "scripts" / "nachlauf_manual_adapters"

SPECS = {
    "rlinear": lambda: nl.UnitSpec(
        unit="rlinear", paper_id="nachlauf_manual/rlinear", claim_id="rlinear_etth1_96_mse",
        claim_text="RLinear reaches MSE 0.366 on ETTh1, horizon 96 (Table 2)", target=0.366, tolerance=0.01,
        repo_url="https://github.com/plumprc/RTSF.git", commit="0fec00104f754f4fbf795b9b4da5fa2459b32e76",
        # script.md:3 nennt `--model Linear`; der Code kennt nur RLinear (Vorversuch): check_command listet diese Abweichung
        command="python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 "
                "--learning_rate 0.005 --model RLinear --rev --seq_len 336 --pred_len 96 --gpu 3",
        command_source={"file": "script.md", "line": 3}, script="run.py",
        seed={"kind": "copy", "repo_seed": 1024, "regex": r"fix_seed = 1024", "template": "fix_seed = {seed}"},
        seed_note="copy of run.py, only the line fix_seed changed (diff checked in the branch)",
        adapter_source=(ADAPTERS / "rtsf_predict.py").read_text(),
        test_loader={"drop_last": True, "batch": 128, "source": "data_provider/data_factory.py:18"},
        argparse_files=["run.py"], requirements_files=[], pip_packages=["pandas", "scikit-learn", "matplotlib"],
    ),
    "dlinear": lambda: nl.UnitSpec(
        unit="dlinear", paper_id="nachlauf_manual/dlinear", claim_id="dlinear_etth1_96_mse",
        claim_text="DLinear reaches MSE 0.375 on ETTh1, horizon 96 (Table 2)", target=0.375, tolerance=0.01,
        repo_url="https://github.com/cure-lab/LTSF-Linear.git", commit="0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
        command="python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear "
                "--data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des Exp --itr 1 --batch_size 32 --learning_rate 0.005",
        command_source={"file": "scripts/EXP-LongForecasting/Linear/etth1.sh", "line": 12}, script="run_longExp.py",
        seed={"kind": "copy", "repo_seed": 2021, "regex": r"fix_seed = 2021", "template": "fix_seed = {seed}"},
        seed_note="copy of run_longExp.py, only the line fix_seed changed (diff checked in the branch)",
        adapter_source=(ADAPTERS / "ltsf_predict.py").read_text(),
        test_loader={"drop_last": False, "batch": 32, "source": "data_provider/data_factory.py:20"},
        argparse_files=["run_longExp.py"], requirements_files=["requirements.txt"], pip_packages=["pandas", "scikit-learn", "matplotlib"],
    ),
}


NEGATIVE_DIR = ADAPTERS / "negative"
# Negativkontrollen (nachlauf_v1.md, Erfolgsregel 2): Einheit fest, Repo-Seed, erwartete ablehnende Pruefung
NEGATIVE = {
    "a": {"unit": "rlinear", "adapter": "a_copy_neighbors_rtsf.py", "expect": {"P2"},
          "what": "adapter copies targets from later windows of the file (window i+96)"},
    "b": {"unit": "dlinear", "adapter": None, "expect": {"P1"}, "what": "start command with --train_only True"},
    "c": {"unit": "rlinear", "adapter": "c_shifted_rtsf.py", "expect": {"P3", "P2"},
          "what": "honest predictions written shifted by one window (cyclic)"},
}
C_VARIANTS = {"shift": "c_shifted_rtsf.py", "permute": "c_permuted_rtsf.py"}


def negative_spec(kind: str, variant: str = "shift") -> "nl.UnitSpec":
    import dataclasses
    n = NEGATIVE[kind]
    spec = SPECS[n["unit"]]()
    spec = dataclasses.replace(spec, paper_id=f"nachlauf_manual/negative_{kind}", claim_id=f"negative_{kind}_etth1_96_mse")
    if kind == "b":
        return dataclasses.replace(spec, command=spec.command + " --train_only True")
    name = C_VARIANTS[variant] if kind == "c" else n["adapter"]
    return dataclasses.replace(spec, adapter_source=(NEGATIVE_DIR / name).read_text())


def negative_report(kind: str, dossier) -> tuple[bool, str]:
    """(abgelehnt durch die erwartete Pruefung, Zeile). Abgelehnt = Versuch ungueltig; Pruefung(en) und Grund aus dem Dossier."""
    at = dossier.attempts[0]
    info = next(iter(dossier.nachlauf.values()))
    checks = sorted({c for v in info["rejected_by"].values() for c in v})
    rejected = at.status == "ungueltig" and bool(checks)
    ok = rejected and bool(set(checks) & NEGATIVE[kind]["expect"])
    line = (f"negative {kind} ({NEGATIVE[kind]['what']}): abgelehnt: {'ja' if rejected else 'nein'} | Pruefung: {'+'.join(checks) or '-'} "
            f"(erwartet: {'/'.join(sorted(NEGATIVE[kind]['expect']))}) | Grund: {at.error_summary or at.status}")
    return ok, line


def default_factory():
    """Echte Sandbox-Fabrik, derselbe Importpfad wie im Betrieb (tests/test_nachlauf_m2.py importiert ihn ohne Client)."""
    from repro_agent.sandbox.base import get_sandbox
    return lambda: get_sandbox("nebius")


def main(argv: list[str] | None = None, sandbox_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=sorted(SPECS))
    ap.add_argument("--negative", choices=sorted(NEGATIVE), help="M2: Negativkontrolle (Einheit, Seed und Adapter fest), Exit-Code 0 nur wenn abgelehnt")
    ap.add_argument("--variant", choices=sorted(C_VARIANTS), default="shift", help="nur --negative c: um ein Fenster verschoben oder zufaellig permutiert")
    ap.add_argument("--seeds", default="repo,0,1,2,3", help="Komma-Liste, `repo` = Seed des Repos")
    ap.add_argument("--offsets", type=int, default=independence.N_OFFSETS, help="Zahl der P2-Versaetze je Seed")
    ap.add_argument("--dry-run", action="store_true", help="nur Befehle drucken, keinen Sandbox-Client erzeugen")
    a = ap.parse_args(argv)
    if bool(a.unit) == bool(a.negative):
        ap.error("genau eines von --unit und --negative angeben")
    spec = negative_spec(a.negative, a.variant) if a.negative else SPECS[a.unit]()
    name = f"negative_{a.negative}" if a.negative else a.unit
    seeds = [spec.repo_seed] if a.negative else [spec.repo_seed if t.strip() == "repo" else int(t) for t in a.seeds.split(",") if t.strip()]
    if a.dry_run:
        if a.negative:
            print(f"# Negativkontrolle {a.negative} ({NEGATIVE[a.negative]['what']}): erwartet abgelehnt durch {'/'.join(sorted(NEGATIVE[a.negative]['expect']))}"
                  + ("; P1 lehnt vor jedem Training ab, die Trainings-Befehle unten laufen dann nicht" if a.negative == "b" else ""))
        print("# Trainings-Snapshot"); [print("  $", c) for c in nl.training_setup_cmds(spec)]
        print("# Vorhersage-Snapshot (ohne Rohdaten)"); [print("  $", c) for c in nl.predict_setup_cmds(spec)]
        for seed in seeds:
            print(f"\n# seed {seed}: Trainings-Branch")
            if seed != spec.repo_seed:
                print(f"  put_files: {nl.SEED_DIR}/{Path(spec.script).name}  (Kopie, nur die Seed-Zeile geaendert)")
            print("  $", nl.train_cmd(spec, None if seed == spec.repo_seed else f"{nl.SEED_DIR}/{Path(spec.script).name}"))
            print("  $", nl.git_cmd())
            print(f"  $ ls -1 {nl.REPO_DIR}/{spec.checkpoint_glob}   (genau eine Datei, get_file)")
            print("# Vorhersage-Branches: put_files Adapter + Checkpoint")
            print("  smoke $", nl.predict_cmd(smoke=True)); print("  voll  $", nl.predict_cmd())
            print("  Harness frisch nach /opt/harness-<zufall>, dann: python <dir>/etth1_eval.py")
            print(f"  P2: {a.offsets} Versaetze, je ein frischer Branch, Probe ersetzt /data/etth1_public.npz")
        return 0
    if sandbox_factory is None:
        sandbox_factory = default_factory()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / "nachlauf_manual" / name / ts
    dossier = nl.run_unit(sandbox_factory(), spec, seeds, offsets=a.offsets, out_dir=out)
    (out / "dossier.json").write_text(__import__("json").dumps(__import__("dataclasses").asdict(dossier), indent=2, ensure_ascii=False))
    if a.negative:
        ok, line = negative_report(a.negative, dossier)
        print(line)
        (out / "dossier.json").write_text(__import__("json").dumps(__import__("dataclasses").asdict(dossier), indent=2, ensure_ascii=False))
        return 0 if ok else 1
    for at in dossier.attempts:
        print(f"{a.unit} seed {at.seed}: {at.status} measured={at.measured} {at.error_summary or ''}"[:300])
    v = dossier.verdicts[0]
    print(f"{a.unit}: {v.urteil} mean={v.measured_mean} std={v.measured_std}\n{out / 'dossier.json'}")
    return 0 if v.urteil in ("reproduziert", "teilweise", "nicht reproduziert") else 1


if __name__ == "__main__":
    sys.exit(main())

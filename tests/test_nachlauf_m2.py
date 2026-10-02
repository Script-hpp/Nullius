"""Nachlauf M2: drei Negativkontrollen (nachlauf_v1.md, Erfolgsregel 2) mit Fake-Sandbox, Rechentest fuer (a), Import-Test der Fabrik."""
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from repro_agent import nachlauf as nl
from repro_agent.harness import independence as ind
from repro_agent.harness.code_check import find_violations
from repro_agent.sandbox.base import RunResult

sys.path.insert(0, str(Path(__file__).parent))
from test_nachlauf import ETTH1_SH, FILES, FakeSandbox, csv_bytes, spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN_UNIT = nl.run_unit


def load_script(rel: str, name: str):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


M = load_script("scripts/run_nachlauf_manual.py", "run_nachlauf_manual_m2")
NEG = ROOT / "scripts" / "nachlauf_manual_adapters" / "negative"
STEPS = 100


def block(fixture: str, kind: str) -> str:
    text = (NEG / fixture).read_text()
    return text.split(f"# --- negative control {kind}: begin\n")[1].split(f"# --- negative control {kind}: end")[0]


def fixture_adapter(fixture: str, kind: str):
    """Der Kern des Fixtures (der eingefuegte Block) auf ehrlichen Vorhersagen; x = test_inputs[:, -STEPS:, :] wie im Adapter."""
    code = block(fixture, kind)

    def adapter(x, scale=0.9):
        pred = honest(x, scale)
        env = {"np": np, "x": x, "pred": pred}
        exec(code, env)   # noqa: S102  (eigener Fixture-Text)
        return env["pred"]
    return adapter


def honest(x, scale=0.9):
    return np.repeat(x[:, -1:, :], 96, axis=1) * scale


A = fixture_adapter("a_copy_neighbors_rtsf.py", "a")
C_SHIFT = fixture_adapter("c_shifted_rtsf.py", "c")
C_PERM = fixture_adapter("c_permuted_rtsf.py", "c")


# ---- synthetisches ETT-aehnliches Signal mit Gleitfenster-Struktur (Fenster j = Zeilen j.., Ziel i = Zeilen i+steps..) ------------

def series_public(n=900, steps=STEPS, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n + steps + 96)[:, None]
    phase = rng.uniform(0, 6.28, size=(1, 7))
    s = np.sin(2 * np.pi * t / 24 + phase) + 0.5 * np.sin(2 * np.pi * t / 168 + phase) + 0.3 * rng.normal(size=(len(t), 7)).cumsum(0) / 20
    s = (s - s.mean(0)) / s.std(0)
    idx = np.arange(n)[:, None] + np.arange(steps)[None, :]
    tgt = np.arange(n)[:, None] + steps + np.arange(96)[None, :]
    pub = {"train": s[:50].astype(np.float32), "val": s[50:80].astype(np.float32), "test_inputs": s[idx].astype(np.float32)}
    return pub, s[tgt]


def mse_all_and_droplast(pred, truth, batch=128):
    k = (len(pred) // batch) * batch
    return float(((pred - truth) ** 2).mean()), float(((pred[:k] - truth[:k]) ** 2).mean())


class SeriesFake(FakeSandbox):
    """Wie FakeSandbox, aber der Harness rechnet den echten MSE aus den geschriebenen Vorhersagen; die Eigenausgabe des Autorencodes ist der
    drop_last-128-MSE der ehrlichen Vorhersagen (wie bei RLinear)."""

    def __init__(self, adapter, **kw):
        pub, self.truth = series_public()
        own = mse_all_and_droplast(honest(pub["test_inputs"]), self.truth)[1]
        kw.setdefault("own", f"mse:{own:.4f}, mae:0.4")
        super().__init__(pub, adapter=adapter, **kw)
        self.own_value = own

    def run(self, b, cmd, timeout_s):
        if cmd.startswith("python /opt/harness-") and cmd.endswith("etth1_eval.py"):
            self.ops.append(("run", b, cmd, timeout_s))
            pred = ind.parse_predictions(self.branches[b]["/output/predictions.csv"])
            mse, dl = mse_all_and_droplast(pred, self.truth)
            out = {"status": "ok", "metrics": {"mse": mse, "mae": 0.4}, "baselines": {}, "metrics_droplast": {"128": {"mse": dl}}, "diagnostics": {}}
            return RunResult(0, json.dumps(out) + "\n", "", False)
        return super().run(b, cmd, timeout_s)


def go(sb, sp=None, offsets=(5, 100, 500, 800)):
    sp = sp or spec(target=round(sb.own_value, 4))
    return REAL_RUN_UNIT(sb, sp, [1024], offsets=list(offsets), public_loader=lambda: sb.pub, np_rng=np.random.default_rng(0), log=lambda *_: None)


def rejected(d):
    return next(iter(d.nachlauf.values()))["rejected_by"].get("1024", [])


# ---- Kontrolle: ehrlicher Adapter besteht (sonst waere jede Ablehnung wertlos) -----------------------------------------------------

def test_honest_adapter_passes_on_the_synthetic_signal():
    d = go(SeriesFake(honest))
    assert d.attempts[0].status == "stabil" and rejected(d) == []
    info = next(iter(d.nachlauf.values()))
    assert info["self_check"]["1024"]["status"] == "geprueft" and info["independence"]["1024"]["ok"]


# ---- (a) Abschreiber -----------------------------------------------------------------------------------------------------------

def test_a_copier_is_rejected_by_p2_and_p3_in_the_flow():
    sb = SeriesFake(A)
    d = go(sb)
    at = d.attempts[0]
    assert at.status == "ungueltig" and "prediction depends on other windows" in at.error_summary
    assert rejected(d) == ["P3", "P2"]                      # beide Pruefungen melden: MSE nahe 0 gegen Eigenausgabe, Proben weichen ab
    assert at.metrics["mse"] < 0.2 * sb.own_value           # im vollen Lauf MSE nahe 0 (Abschreiber sieht das Ziel)
    ok, line = M.negative_report("a", d)
    assert ok and "abgelehnt: ja" in line and "P2" in line


def test_a_numerics_on_real_shapes_without_sandbox():
    n, steps = 2785, 336
    pub, truth = series_public(n=n, steps=steps)
    x = pub["test_inputs"]
    full = A(x)
    mse_h, _ = mse_all_and_droplast(honest(x), truth)
    mse_c, _ = mse_all_and_droplast(full, truth)
    assert mse_c < 0.06 * mse_h                             # nur die letzten 96 von 2785 Fenstern sind ehrlich: MSE nahe 0
    assert np.allclose(full[: n - 96], truth[: n - 96], atol=1e-6)
    bad = []
    for r in range(ind.STRIDE):                             # jeder der 816 moeglichen Versaetze
        arrays, mapping = ind.make(pub, r, np.random.default_rng(r))
        cmp = ind.compare(full, A(arrays["test_inputs"]), mapping)   # Probe: kein Nachbar i+96, Rueckfall auf das Modell
        if cmp["ok"] or not cmp["max_abs_diff"] > 1e-4:
            bad.append(r)
    assert bad == []


# ---- (b) verbotener Startbefehl ------------------------------------------------------------------------------------------------

RUN_LONGEXP_HELP = ("parser.add_argument('--train_only', type=bool, required=False, default=False, "
                    "help='perform training on full input dataset without validation and testing')\n")


def test_b_train_only_rejected_by_p1_before_any_training():
    sp = M.negative_spec("b")
    assert sp.command.endswith("--train_only True") and sp.unit == "dlinear"
    r = nl.check_command(sp.command, {"run_longExp.py": RUN_LONGEXP_HELP, sp.command_source["file"]: ETTH1_SH}, sp.command_source)
    assert not r["valid"] and any("--train_only" in h for h in r["forbidden"])
    # Flow (Fake): Abbruch vor dem Training; es laeuft kein Befehl mit unshare, der einzige Trainings-Snapshot-Branch liest nur Dateien
    sb = FakeSandbox(series_public()[0])
    d = nl.run_unit(sb, spec(command=spec().command + " --train_only True"), [1024], offsets=[5], public_loader=lambda: sb.pub, log=lambda *_: None)
    assert d.attempts[0].status == "ungueltig" and rejected(d) == ["P1"]
    assert not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)
    train_runs = [o[2] for o in sb.ops if o[0] == "run" and sb.kind[o[1]] == "train"]
    assert all(c.startswith(("cat ", "pip freeze", "git -C")) for c in train_runs)
    ok, line = M.negative_report("b", d)
    assert ok and "Pruefung: P1" in line


def test_help_text_pattern_catches_the_real_train_only_help_under_another_name():
    hits = nl.forbidden_flags("python run.py --use_full", {"run.py": RUN_LONGEXP_HELP.replace("--train_only", "--use_full")})
    assert hits and "--use_full" in hits[0]


# ---- (c) vertauschte Fenster ---------------------------------------------------------------------------------------------------

def test_c_shift_by_one_window_is_rejected_by_p2_always():
    sb = SeriesFake(C_SHIFT)
    d = go(sb)
    checks = rejected(d)
    assert d.attempts[0].status == "ungueltig" and "P2" in checks   # P2 greift strukturell: in der Probe fehlt der Vorgaenger
    ok, line = M.negative_report("c", d)
    assert ok and "abgelehnt: ja" in line


def test_c_shift_p3_depends_on_the_signal_and_on_an_own_output():
    """P3 greift nur, wenn der Autorencode eine eigene Metrik ausgibt UND die Verschiebung den MSE um > 0,001 veraendert. Hier zur Dokumentation."""
    sb = SeriesFake(C_SHIFT)
    d = go(sb)
    sc = next(iter(d.nachlauf.values()))["self_check"]["1024"]
    assert sc["distance"] > 0 and (sc["status"] == "ungueltig") == ("P3" in rejected(d))
    # ohne Eigenausgabe: P3 nicht geprueft, P2 allein lehnt ab
    d2 = go(SeriesFake(C_SHIFT, own="no metric printed"))
    assert rejected(d2) == ["P2"] and next(iter(d2.nachlauf.values()))["self_check"]["1024"]["status"] == "nicht geprueft"


def test_c_random_permutation_rejected_by_p3_and_p2():
    d = go(SeriesFake(C_PERM))
    assert rejected(d) == ["P3", "P2"]
    ok, line = M.negative_report("c", d)
    assert ok and "P2+P3" in line


# ---- Fixtures selbst -----------------------------------------------------------------------------------------------------------

def test_fixtures_pass_static_check_and_differ_from_the_honest_adapter_only_in_the_marked_block():
    honest_src = (ROOT / "scripts" / "nachlauf_manual_adapters" / "rtsf_predict.py").read_text()
    body = lambda t: t.split('"""', 2)[2]  # noqa: E731
    for fx, kind in [("a_copy_neighbors_rtsf.py", "a"), ("c_shifted_rtsf.py", "c"), ("c_permuted_rtsf.py", "c")]:
        src = (NEG / fx).read_text()
        assert find_violations(src) == [], fx                    # getestet wird P2/P3, nicht die statische Pruefung
        stripped = body(src).replace(f"# --- negative control {kind}: begin\n{block(fx, kind)}# --- negative control {kind}: end\n", "")
        assert stripped == body(honest_src), fx
    assert "x[96:, -96:, :]" in block("a_copy_neighbors_rtsf.py", "a")


def test_negative_cli_dry_run_creates_no_client(capsys):
    def boom():
        raise AssertionError("kein Client im Dry-Run")
    for kind in ("a", "b", "c"):
        assert M.main(["--negative", kind, "--dry-run"], sandbox_factory=boom) == 0
    assert M.main(["--negative", "c", "--variant", "permute", "--dry-run"], sandbox_factory=boom) == 0
    out = capsys.readouterr().out
    assert "erwartet abgelehnt durch P1" in out and "unshare -n" in out
    with pytest.raises(SystemExit):
        M.main(["--dry-run"])


def test_negative_exit_code_only_when_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "ROOT", tmp_path)
    sb = SeriesFake(honest)
    def fake_run(box):
        def run(*a, **k):
            k["out_dir"].mkdir(parents=True, exist_ok=True)
            return go(box)
        return run
    monkeypatch.setattr(nl, "run_unit", fake_run(sb))
    assert M.main(["--negative", "c"], sandbox_factory=lambda: sb) == 1      # ehrlicher Adapter: nicht abgelehnt -> Exit 1
    sb2 = SeriesFake(C_PERM)
    monkeypatch.setattr(nl, "run_unit", fake_run(sb2))
    assert M.main(["--negative", "c"], sandbox_factory=lambda: sb2) == 0


# ---- Import-Test der echten Sandbox-Fabrik (ohne Client) -------------------------------------------------------------------------

@pytest.mark.parametrize("rel", ["scripts/run_nachlauf_manual.py", "scripts/prestudy_nachlauf/sandbox_run.py"])
def test_scripts_import_the_real_sandbox_factory_path(rel):
    """Die Fake-Tests uebergeben eine eigene Fabrik und fangen einen falschen Modulnamen nicht (Fehler in M1: `from repro_agent.sandbox import get_sandbox`).
    Hier laeuft die echte Importzeile der Skripte; erzeugt wird kein Client (kein Netz, keine Zugangsdaten)."""
    mod = load_script(rel, "imp_" + Path(rel).stem)
    factory = mod.default_factory()
    assert callable(factory)
    from repro_agent.sandbox import base
    assert callable(base.get_sandbox)

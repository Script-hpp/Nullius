"""Detektor "letzter Test-Batch verworfen" (ETTh1): Rechnung, Hinweisregel, Harness-JSON, kein Einfluss aufs Urteil, kein Prompt-Pfad. Attrappen, kein Netz."""

import json
import re
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import pytest

from repro_agent.dossier import Claim, Dossier
from repro_agent.droplast_note import NEAR_FRACTION, droplast_note, droplast_text
from repro_agent.harness import droplast
from repro_agent.report import render_markdown
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import etth1_task
from test_loop_fakes import FakeSandbox, make_orch

HARNESS = Path(__file__).resolve().parents[1] / "src" / "repro_agent" / "harness"


def row(mse, dropped=97, total=2785):
    return {"mse": mse, "mae": mse + 0.03, "windows": total - dropped, "dropped": dropped}


# ---- droplast.py ----

def test_kept_windows_and_dropped_counts_for_2785():
    assert [droplast.kept_windows(2785, b) for b in (32, 64, 128, 256)] == [2784, 2752, 2688, 2560]
    assert [2785 - droplast.kept_windows(2785, b) for b in (32, 64, 128, 256)] == [1, 33, 97, 225]


def test_metrics_droplast_uses_leading_windows_only():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(300, 4, 2))
    pred = truth.copy()
    pred[256:] += 5.0                                    # Fehler nur in den letzten 44 Fenstern
    out = droplast.metrics_droplast(pred, truth, batches=(32, 128, 256, 512))
    assert out["256"]["mse"] == 0.0 and out["256"]["windows"] == 256 and out["256"]["dropped"] == 44
    assert out["128"]["mse"] == 0.0 and out["128"]["dropped"] == 300 - 256       # 2 volle Batches = 256 Fenster
    assert out["32"]["dropped"] == 300 - 288 and out["32"]["mse"] > 0     # Fenster 256..287 gehoeren noch dazu
    assert "512" not in out                                               # nichts bliebe uebrig
    full = float(((pred - truth) ** 2).mean())
    assert full > out["32"]["mse"] > 0 and set(out["256"]) == {"mse", "mae", "windows", "dropped"}


# ---- Hinweisregel ----

def test_rule_fires_when_claim_hits_droplast_value_not_full():
    # RLinear-aehnlich: Vollwert 0,3704, drop_last(128) 0,3665, Behauptung 0,366, Toleranz 0,01 (Schwelle tol/4 = 0,0025)
    per_seed = [{"128": row(0.3665), "32": row(0.3703, 1), "256": row(0.3650, 225)}] * 2
    n = droplast_note("mse", 0.366, 0.01, 0.3704, per_seed)
    assert n and n["batch_size"] == 128 and n["kind"] == "within_tolerance"
    assert n["windows_dropped"] == 97 and n["windows_total"] == 2785 and n["n_seeds"] == 2
    assert n["sources"] and n["batch_context"]


def test_rule_outside_tolerance_kind():
    n = droplast_note("mse", 0.350, 0.005, 0.3704, [{"128": row(0.3510)}])
    assert n and n["kind"] == "outside_tolerance"
    assert "may explain the difference" in droplast_text(n) and "within tolerance" not in droplast_text(n)


def test_rule_silent_when_full_value_is_closer():
    # SparseTSF-aehnlich: Behauptung 0,362, Vollwert 0,3607 (Abstand 0,0013), drop_last(128) 0,3587 (Abstand 0,0033)
    per_seed = [{"32": row(0.3606, 1), "64": row(0.3600, 33), "128": row(0.3587), "256": row(0.3570, 225)}]
    assert droplast_note("mse", 0.362, 0.01, 0.3607, per_seed) is None


def test_rule_silent_when_full_value_already_hits_the_claim():
    # lokale Pruefung: SparseTSF Vollwert 0,3620, drop_last(32) 0,3620 (1 Fenster), Behauptung 0,362; NLinear 0,3746 gegen 0,3735 bei 0,374
    assert droplast_note("mse", 0.362, 0.01, 0.36201, [{"32": row(0.36200, 1)}]) is None
    assert droplast_note("mse", 0.374, 0.01, 0.3746, [{"64": row(0.3735, 33)}]) is None
    just_over = 0.366 + NEAR_FRACTION * 0.01 * 1.001                       # Vollwert knapp ausserhalb von tol/4: Hinweis moeglich
    assert droplast_note("mse", 0.366, 0.01, just_over, [{"128": row(0.3665)}]) is not None


def test_rule_threshold_is_quarter_tolerance_exactly():
    tol, claim, full = 0.01, 0.40, 0.42
    at = {"128": row(claim - NEAR_FRACTION * tol * 0.999)}
    over = {"128": row(claim - NEAR_FRACTION * tol * 1.001)}
    assert droplast_note("mse", claim, tol, full, [at]) is not None
    assert droplast_note("mse", claim, tol, full, [over]) is None


def test_rule_other_metric_or_empty_gives_none():
    assert droplast_note("accuracy", 0.8, 0.01, 0.81, [{"128": row(0.8)}]) is None
    assert droplast_note("mse", 0.4, 0.01, 0.42, []) is None
    assert droplast_note("mse", 0.4, 0.0, 0.42, [{"128": row(0.4)}]) is None


def test_text_within_tolerance_says_so():
    n = droplast_note("mse", 0.366, 0.01, 0.3704, [{"128": row(0.3665)}])
    t = droplast_text(n)
    assert "last 97 of 2785 test windows are skipped" in t and "still within tolerance" in t
    assert "0.3704" in t and "0.3665" in t and "not a criticism" in t


# ---- Harness-JSON (etth1_eval.py), Hauptmetrik unveraendert ----

def test_harness_json_has_main_metric_on_all_windows_plus_droplast(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(HARNESS))
    try:
        import etth1_data
        n_win, h, c = 300, etth1_data.PRED_LEN, etth1_data.N_CHANNELS
        truth = np.random.default_rng(1).normal(size=(n_win, h, c))
        monkeypatch.setattr(etth1_data, "n_windows", lambda pl=h: n_win)
        monkeypatch.setattr(etth1_data, "download", lambda p: p / "x.csv")
        monkeypatch.setattr(etth1_data, "targets", lambda csv, pl=h: truth)
        monkeypatch.setattr(etth1_data, "baselines", lambda csv, pl=h: {})
        pred = truth + np.random.default_rng(2).normal(scale=0.1, size=truth.shape)
        pred[256:] += 3.0
        pcsv = tmp_path / "predictions.csv"
        w, hh, cc = np.meshgrid(np.arange(n_win), np.arange(h), np.arange(c), indexing="ij")
        pcsv.write_text("id,prediction\n" + "\n".join(f"{a}_{b}_{d},{v}" for a, b, d, v in zip(w.ravel(), hh.ravel(), cc.ravel(), pred.ravel())) + "\n")
        src = (HARNESS / "etth1_eval.py").read_text().replace('"/output/predictions.csv"', repr(str(pcsv)))
        exec(compile(src, "etth1_eval.py", "exec"), {"__name__": "etth1_eval", "__file__": str(HARNESS / "etth1_eval.py")})
    finally:
        sys.path.remove(str(HARNESS))
        for m in ("etth1_data", "droplast", "metrics", "diagnostics"):
            sys.modules.pop(m, None)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    full = float(((pred - truth) ** 2).mean())
    assert out["status"] == "ok" and out["metrics"]["mse"] == pytest.approx(full, rel=1e-6)    # Hauptmetrik: alle Fenster
    assert out["metrics_droplast"]["256"]["windows"] == 256 and out["metrics_droplast"]["256"]["mse"] < 0.02
    assert out["metrics_droplast"]["256"]["mse"] < out["metrics"]["mse"]


# ---- Loop: Hinweis im Dossier, Urteil unveraendert, nichts im Prompt ----

class EttSandbox(FakeSandbox):
    """ETTh1-Auswertung mit festem Vollwert und drop_last-Wert; merkt sich, dass der Harness die Zusatzwerte liefert."""

    def __init__(self, full, dl, **kw):
        super().__init__(**kw)
        self.full, self.dl = full, dl

    def get_file(self, branch, path):   # Formpruefung: 8 Fenster x 96 Schritte x 7 Kanaele
        return ("id,prediction\n" + "".join(f"{w}_{h}_{c},0.1\n" for w in range(8) for h in range(96) for c in range(7))).encode()

    def run(self, branch, cmd, timeout_s):
        if "etth1_eval.py" in cmd:
            out = {"status": "ok", "metrics": {"mse": self.full, "mae": self.full + 0.03}, "baselines": {},
                   "metrics_droplast": {"128": row(self.dl)}}
            return RunResult(0, json.dumps(out), "", False)
        return super().run(branch, cmd, timeout_s)


def run_ett(tmp_path, full, dl, target=0.366, seeds=3):
    sb = EttSandbox(full, dl)
    o, calls = make_orch(sb, seeds=seeds)
    prompts = []
    o.llm = type("L", (), {"total_cost_usd": 0.0, "gross": staticmethod(lambda x: x * 1.19)})()
    claim = Claim("c", "RLinear 0,366", "mse", target, 0.01, "Tabelle", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": seeds}, claim, etth1_task(), "snap", d, tmp_path)
    return d, claim, sb


def test_loop_writes_note_and_keeps_verdict(tmp_path):
    d, claim, sb = run_ett(tmp_path, 0.3704, 0.3665)
    assert d.verdicts[0].urteil == "reproduziert"                    # Abstand 0,0044 <= Toleranz, unveraendert
    assert d.verdicts[0].measured_mean == pytest.approx(0.3704)      # Urteil beruht auf allen Fenstern
    n = d.droplast_notes["c"]
    assert n["batch_size"] == 128 and n["kind"] == "within_tolerance" and n["full"] == pytest.approx(0.3704)
    assert all(a.metrics_droplast for a in d.attempts if a.measured is not None)
    assert "still within tolerance" in render_markdown(d)
    f = tmp_path / "dossier.json"
    f.write_text(json.dumps(asdict(d)))
    assert Dossier.load(f).droplast_notes == d.droplast_notes


def test_loop_without_hit_has_no_note_and_same_verdict(tmp_path):
    d, _, _ = run_ett(tmp_path, 0.3607, 0.3587, target=0.362)
    assert d.verdicts[0].urteil == "reproduziert" and "c" not in d.droplast_notes
    assert "drop_last" not in render_markdown(d)


def test_note_does_not_depend_on_verdict_outcome(tmp_path):
    a, _, _ = run_ett(tmp_path, 0.3704, 0.3665)
    b, _, _ = run_ett(tmp_path, 0.3704, 0.3665, target=0.30)         # Placebo: Urteil anders, Hinweis aus der Regel
    assert a.verdicts[0].urteil == "reproduziert" and b.verdicts[0].urteil == "nicht reproduziert"
    assert "c" not in b.droplast_notes


def test_no_prompt_path_reads_droplast_fields():
    """Testwerte bleiben beim Harness: die Felder kommen nur in Harness, Dossier, Hinweisregel, Bericht und in loop.py an genau vier Stellen vor."""
    src = Path(__file__).resolve().parents[1] / "src" / "repro_agent"
    fields = ("metrics_droplast", "droplast_note", "_droplast_round")
    owners = {"droplast_note.py", "dossier.py", "report.py", "loop.py", "nachlauf.py", "nachlauf_agent.py", "objection_experiments.py"}   # objection_experiments.py: protocol_change liest sie nur fuer die Regel, nie in einem Prompt (unten geprueft); Modus Nachlauf: nachlauf.py (M1, kein LLM), nachlauf_agent.py (M3, Prompts ohne Testwerte: tests/test_nachlauf_agent.py)
    for f in src.glob("*.py"):
        if any(x in f.read_text() for x in fields):
            assert f.name in owners, f.name
    lines = [l.strip() for l in (src / "loop.py").read_text().splitlines() if any(x in l for x in fields)]
    # Import, Ergebnis-Dict, Aufruf nach dem Urteil, Methode (Kopf, Hinweis, Lesen, Schreiben), Attempt-Feld: keine Zeile baut einen Prompt
    assert not [l for l in lines if re.search(r"prompt|system=|user=|\bask\(|\.chat\(|messages", l)]
    oe = (src / "objection_experiments.py").read_text()
    for name in ("classify_prompt", "_run_parameter_change", "classify_finding"):
        body = oe.split(f"def {name}")[1].split("\ndef ")[0]
        assert "droplast" not in body.lower().replace("drop_last", ""), name
    d = Dossier(paper_id="p", claims=[Claim("c", "t", "mse", 0.3, 0.01, "s", "paper")])
    d.droplast_notes["c"] = {"droplast": 0.123456}
    assert "0.123456" not in d.condensed_for_reasoning() and "droplast" not in d.condensed_for_reasoning().lower()

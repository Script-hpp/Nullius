"""G1: Entscheidungsregel, Aggregat mit G0-Zahlen, Nebius-Huelle (Attrappe), Steckbrief-Neuberechnung ohne Extraktor. Kein Netz, kein Geld."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "review"))

import cg_gate as g0
import cg_gate_g1 as g1


def S(unit, arm, urteil, temp=0.2, missing=False):
    return g0.Sample(unit, arm, temp, urteil=urteil, n_seeds=5, code_missing=missing)


def cells(unit, arm, verdicts):
    return [S(unit, arm, v, t) for v, t in zip(verdicts, g0.TEMPS)]


REF = lambda u: g0.UNITS[u].ref_verdict
BAD = "nicht reproduziert"


def g0_samples(cg_other=8, heute_other=4, kipf=(1, 0), sgcn_heute=0):
    """G0-Zahlen: auf den vier Nicht-GAT-Entwicklungseinheiten `cg_other` bzw. `heute_other` Treffer, dazu Kipf und Shchur-GCN heute."""
    out = []
    others = [u for u in g0.FAIL_UNITS if u not in g1.G1_GAT]
    for arm, total in (("cg", cg_other), ("heute", heute_other)):
        left = total
        for u in others:
            n = min(3, left)
            left -= n
            out += cells(u, arm, [REF(u)] * n + [BAD] * (3 - n))
    out += cells("kipf_gcn_2017", "cg", [REF("kipf_gcn_2017")] * kipf[0] + [BAD] * (3 - kipf[0]))
    out += cells("kipf_gcn_2017", "heute", [REF("kipf_gcn_2017")] * kipf[1] + [BAD] * (3 - kipf[1]))
    out += cells("shchur_pitfalls_2018", "heute", [REF("shchur_pitfalls_2018")] * sgcn_heute + [BAD] * (3 - sgcn_heute))
    return out


def g1_samples(gat_cg=(1, 0), gat_heute=(0, 0), sgcn=3, rlin=3, missing_cg=0):
    out = []
    for u, c, h in zip(g1.G1_GAT, gat_cg, gat_heute):
        out += cells(u, "cg", [REF(u)] * c + [BAD] * (3 - c))
        out += cells(u, "heute", [REF(u)] * h + [BAD] * (3 - h))
    out += cells("shchur_pitfalls_2018", "cg", ["reproduziert"] * sgcn + [BAD] * (3 - sgcn))
    out += cells("rlinear_2023", "cg", ["reproduziert"] * rlin + [BAD] * (3 - rlin))
    for s in out[:missing_cg]:
        s.code_missing = True
    return out


def test_cells_cover_gat_both_arms_and_two_nonreg_units_cg_only():
    assert len(g1.G1_CELLS) == 6 and len(g1.G1_CELLS) * len(g0.TEMPS) == 18
    assert {u for u, a in g1.G1_CELLS if a == "heute"} == set(g1.G1_GAT)
    assert {u for u, a in g1.G1_CELLS if a == "cg"} == set(g1.G1_GAT) | set(g1.G1_NONREG)


def test_one_hit_lead_on_gat_passes_rule_a():
    d = g1.decide_g1(g1_samples(gat_cg=(1, 0), gat_heute=(0, 0)), g0_samples())
    assert d["a_hits_dev_cg"] == 9 and d["a_hits_dev_heute"] == 4 and d["a_pass"] and d["decision"] == "bestanden"


def test_no_lead_fails_rule_a():
    d = g1.decide_g1(g1_samples(gat_cg=(1, 0), gat_heute=(1, 0)), g0_samples())
    assert not d["a_pass"] and d["decision"] == "nicht bestanden"
    d = g1.decide_g1(g1_samples(gat_cg=(0, 0)), g0_samples())
    assert not d["a_pass"]


def test_g0_numbers_are_taken_from_the_rerun_free_units_only():
    d = g1.decide_g1(g1_samples(gat_cg=(2, 1), gat_heute=(0, 1)), g0_samples(cg_other=8, heute_other=4))
    assert d["a_hits_dev_cg"] == 11 and d["a_hits_dev_heute"] == 5   # 8 + 3, 4 + 1


def test_nonregression_each_rerun_unit_needs_two_of_three():
    assert g1.decide_g1(g1_samples(sgcn=1), g0_samples())["decision"] == "nicht bestanden"   # b2
    assert g1.decide_g1(g1_samples(rlin=1), g0_samples())["c_pass"] is False                 # Regel (c) unveraendert
    assert g1.decide_g1(g1_samples(sgcn=2, rlin=2), g0_samples())["decision"] == "bestanden"


def test_rule_b_uses_g0_for_kipf_and_g1_for_shchur_gcn():
    d = g1.decide_g1(g1_samples(sgcn=3), g0_samples(kipf=(1, 0), sgcn_heute=0))
    assert d["b_nonreg_cg"] == 4 and d["b_nonreg_heute"] == 0 and d["b_pass"]


def test_rule_d_counts_g1_calls_for_rerun_units_and_g0_for_the_rest():
    g = g0_samples()
    next(s for s in g if s.unit == "wu_sgc_2019" and s.arm == "cg").code_missing = True
    d = g1.decide_g1(g1_samples(missing_cg=2), g)
    assert d["d_cg_code_calls_missing"] == 3 and not d["d_pass"] and d["decision"] == "nicht bestanden"
    g = g0_samples() + [S("rlinear_2023", "cg", "nicht reproduzierbar (Setup)", missing=True)]   # veraltete G0-Probe einer erneut gemessenen Einheit
    assert g1.decide_g1(g1_samples(), g)["d_cg_code_calls_missing"] == 0


def test_incomplete_until_every_cell_has_three_samples():
    d = g1.decide_g1(g1_samples()[:-1], g0_samples())
    assert d["complete"] is False and d["decision"] == "unvollstaendig"


def test_report_lists_g1_and_g0_hits_side_by_side():
    text = g1.report(g1_samples(), g0_samples(), 0.1234)
    assert "| velickovic_gat_2018 | heute | 0/3 |" in text and "Kosten G1" in text and '"decision": "bestanden"' in text


class FakeInner:
    def __init__(self):
        self._sessions, self.runs, self.n = {}, [], 0
        self.tags = {"repro-snapshot-have"}

    def base_snapshot(self, image, setup_cmds, label=None):
        return f"built-{label}"

    def branch(self, snapshot):
        if snapshot.startswith("repro-snapshot-") and snapshot not in self.tags:
            raise RuntimeError("tag not found")
        self.n += 1
        name = f"b{self.n}"
        self._sessions[name] = object()
        return name

    def run(self, branch, cmd, timeout_s):
        self.runs.append((branch, cmd, timeout_s))
        return SimpleNamespace(exit_code=0, stdout="", stderr="", timed_out=False)

    def put_file(self, *a):
        pass

    put_files = get_file = put_file


def _sandbox(monkeypatch):
    import gate_sandbox
    import repro_agent.sandbox.nebius as neb
    monkeypatch.setattr(neb, "NebiusSandbox", FakeInner)
    return gate_sandbox.NebiusGateSandbox(max_parallel=2)


def test_nebius_sandbox_reuses_existing_snapshot_tag_and_builds_otherwise(monkeypatch):
    sb = _sandbox(monkeypatch)
    assert sb.base_snapshot("img", [], label="have") == "repro-snapshot-have"
    assert sb.inner._sessions == {}                    # Sondierungs-Branch wurde vergessen
    assert sb.base_snapshot("img", [], label="gone") == "built-gone"


def test_nebius_sandbox_times_training_and_releases_branches(monkeypatch):
    sb = _sandbox(monkeypatch)
    snap = sb.base_snapshot("img", [], label="have")
    sb.begin_attempt()
    b = sb.branch(snap)
    sb.run(b, "cd /work && SEED=0 SMOKE=1 unshare -n python train.py", 90)
    assert b not in sb.times                          # Smoke-Lauf zaehlt nicht als Trainingszeit
    sb.run(b, "cd /work && SEED=0 unshare -n timeout -s ABRT 585 python train.py", 600)
    assert b in sb.times and sb.inner.runs[-1][1].startswith("cd /work") and "unshare -n" in sb.inner.runs[-1][1]   # Netzsperre bleibt im Kommando
    assert sb.for_unit("x") is sb
    sb.release_attempt()
    assert b not in sb.inner._sessions


def test_nebius_sandbox_limits_parallel_runs(monkeypatch):
    sb = _sandbox(monkeypatch)
    active, peak, lock = [0], [0], threading.Lock()
    orig = sb.inner.run

    def slow(branch, cmd, timeout_s):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        import time
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return orig(branch, cmd, timeout_s)

    sb.inner.run = slow
    ts = [threading.Thread(target=sb.run, args=("b", "x", 1)) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] <= 2

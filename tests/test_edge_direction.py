"""B2 (v5 Schritt B): Kanten in beiden Richtungen, Black-Box-Probe auf synthetischem Graphen. Attrappen, kein Netz, kein Geld."""

import re

import numpy as np

from repro_agent.harness.edge_direction import (
    BOTH, ONE, UNCLEAR, NOT_MEASURABLE, classify_edge_direction, make_probe, score_probe)
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_lcc_task, cora_task, etth1_task
from test_graph_repair import V2, h, run, setup, types
from test_loop_fakes import FakeSandbox

ONEDIR = "print('ONEDIR')\n"       # Attrappen-Code, der Kanten nur in einer Richtung nutzt
TWODIR = "print('TWODIR')\n"


def test_classify_thresholds():
    assert classify_edge_direction(0.9, 0.9) == BOTH and classify_edge_direction(0.9, 0.61) == BOTH
    assert classify_edge_direction(0.9, 0.6) == ONE                          # Abstand genau 0,30
    assert classify_edge_direction(0.9, 0.14) == ONE and classify_edge_direction(0.14, 0.9) == ONE
    assert classify_edge_direction(0.2, 0.3) == UNCLEAR                      # beide schlecht: Sache der Ablation
    assert classify_edge_direction(None, 0.9) == NOT_MEASURABLE


def test_probe_structure():
    for t, (n, nv, nt) in ((cora_task(), (2708, 500, 1000)), (cora_lcc_task(), (2485, 210, 2135))):
        assert t.edge_probe[1:] == (n, nv, nt)
        npz, truth = make_probe(n, nv, nt, seed=0)
        assert npz["features"].shape == (n, 1433) and npz["labels"].shape == (n,)
        e = npz["edges"]
        pairs = {tuple(x) for x in e.tolist()}
        assert all((b, a) in pairs for a, b in pairs)                         # jede Kante in beiden Richtungen
        assert len(npz["train_idx"]) == 140 and set(np.bincount(npz["labels"][npz["train_idx"]])) == {20}
        assert (npz["labels"][npz["test_idx"]] == -1).all()                    # Testlabels verborgen
        assert not set(npz["train_idx"]) & set(npz["val_idx"]) and not set(npz["test_idx"]) & set(npz["train_idx"])
        assert len(truth["side_a"]) > 0 and len(truth["side_b"]) > 0
    npz2, _ = make_probe(2708, 500, 1000, seed=0)
    assert np.array_equal(npz["features"].shape, (2485, 1433)) and np.array_equal(npz2["edges"], make_probe(2708, 500, 1000, seed=0)[0]["edges"])
    assert etth1_task().edge_probe is None


def test_own_features_carry_no_own_label_signal():
    """Das eigene Label steckt nur in den Merkmalen des Partners (Signalwoerter), nicht in den eigenen."""
    npz, truth = make_probe(2708, 500, 1000, seed=0)
    feats, labels, m = npz["features"], truth["labels"], 2708 // 2
    sig = lambda i: int(np.argmax([feats[i, c * 20:(c + 1) * 20].sum() for c in range(7)]))
    own = np.mean([sig(i) == labels[i] for i in range(m)])
    partner = np.mean([sig(i + m) == labels[i] for i in range(m)])
    assert own < 0.3 and partner > 0.9


class ProbeSandbox(FakeSandbox):
    """Probe-Branch (synthetische npz) bekommt Vorhersagen je nach Code: TWODIR beide Seiten richtig, ONEDIR nur Seite A."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.code, self.probe, self.preds = {}, set(), {}

    def put_file(self, branch, path, content):
        if path.endswith("train.py"):
            self.code[branch] = content.decode()
        if path.endswith("_public.npz"):
            self.probe.add(branch)
        super().put_file(branch, path, content)

    def run(self, branch, cmd, timeout_s):
        if branch in self.probe and "train.py" in cmd:
            _, truth = make_probe(*cora_task().edge_probe[1:], seed=0)
            rng = np.random.default_rng(1)
            rows = []
            for i in sorted(truth["side_a"] | truth["side_b"]):
                ok = "TWODIR" in self.code[branch] or i in truth["side_a"]
                rows.append((i, int(truth["labels"][i]) if ok else int((truth["labels"][i] + 1 + rng.integers(0, 6)) % 7)))
            self.preds[branch] = ("id,prediction\n" + "".join(f"{i},{p}\n" for i, p in rows)).encode()
            self.cmds.append((branch, cmd))
            return RunResult(0, "", "", False)
        return super().run(branch, cmd, timeout_s)

    def get_file(self, branch, path):
        if branch in self.probe and path == "/output/predictions.csv":
            return self.preds[branch]
        return super().get_file(branch, path)


def test_probe_scoring_and_isolated_run(tmp_path):
    o, calls, events = setup([0.83], [0.5], write=TWODIR, sandbox_cls=ProbeSandbox)
    d = run(o, tmp_path)
    (ev,) = types(events, "edge_direction_check")
    assert ev["verdict"] == BOTH and ev["acc_side_a"] == 1.0 and ev["acc_side_b"] == 1.0
    assert calls["repair"] == [] and d.edge_direction["c"][0]["verdict"] == BOTH
    probe_cmds = [c for b, c in o.sandbox.cmds if b in o.sandbox.probe]
    assert probe_cmds and all("unshare -n" in c for c in probe_cmds)           # ohne Netz wie jeder Trainingslauf


def test_one_direction_triggers_repair_with_fixed_hint(tmp_path):
    o, calls, events = setup([0.83], [0.5], repairs=[TWODIR], write=ONEDIR, sandbox_cls=ProbeSandbox)
    d = run(o, tmp_path)
    (first, second) = types(events, "edge_direction_check")
    assert first["verdict"] == ONE and second["verdict"] == BOTH
    (msg,) = calls["repair"]
    assert "both directions of every edge" in msg and not re.search(r"\d", msg.replace("u < v", ""))   # keine Zahl im Hinweis
    (gr,) = types(events, "graph_repair")
    assert gr["kind"] == "edge_direction" and gr["from_hash"] == h(ONEDIR) and gr["to_hash"] == h(TWODIR)
    assert d.graph_repairs == {"c": 1} and {a.code_hash for a in d.attempts if a.seed > 0} == {h(TWODIR)}


def test_edge_repairs_share_graph_budget(tmp_path):
    o, calls, events = setup([0.83], [0.5], repairs=[ONEDIR + "# 2\n", ONEDIR + "# 3\n"], write=ONEDIR, sandbox_cls=ProbeSandbox)
    d = run(o, tmp_path)
    assert len(calls["repair"]) == 2 and d.graph_repairs == {"c": 2}           # MAX_GRAPH_REPAIRS


def test_no_probe_without_flag_or_for_unused_graph(tmp_path):
    o, calls, events = setup([0.83], [0.5], write=ONEDIR, flag=False, sandbox_cls=ProbeSandbox)
    run(o, tmp_path)
    assert types(events, "edge_direction_check") == []
    o, calls, events = setup([0.83], [0.83], write=ONEDIR, sandbox_cls=ProbeSandbox)   # Graph nicht genutzt: Sache der Ablation
    run(o, tmp_path)
    assert types(events, "edge_direction_check") == []

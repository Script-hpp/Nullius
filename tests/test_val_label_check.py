"""B1 (v5 Schritt B): Validierungslabels im Training. Attrappen, kein Netz, kein Geld."""

import json
from pathlib import Path

import pytest

from repro_agent.harness.val_use_check import classify_val_mask, static_val_use
from repro_agent.tasks import cora_lcc_task, cora_task, etth1_task
from test_candidates import C0, C1, C2, ValSandbox, run, setup

REF = Path(__file__).parent / "reference"

LEAK_CAT = """
import numpy as np, torch
import torch.nn.functional as F
d = np.load('/data/cora_public.npz')
train_idx = torch.from_numpy(d['train_idx']); val_idx = torch.from_numpy(d['val_idx'])

def train_model(model, X, y, idx, opt):
    for _ in range(10):
        opt.zero_grad()
        loss = F.cross_entropy(model(X)[idx], y[idx])
        loss.backward()
        opt.step()

idx = torch.cat([train_idx, val_idx])
train_model(m, X, y, idx, opt)
"""

LEAK_MASK = """
import torch.nn.functional as F
def masked_loss(logits, labels):
    mask = labels >= 0
    return F.cross_entropy(logits[mask], labels[mask])

def run(model, X, labels, opt):
    loss = masked_loss(model(X), labels)
    loss.backward()
"""

SELECTION_ONLY = """
import numpy as np, torch
import torch.nn.functional as F
d = np.load('/data/cora_public.npz')
tr, va = torch.from_numpy(d['train_idx']), torch.from_numpy(d['val_idx'])
num_classes = int(y[y >= 0].max()) + 1
for epoch in range(200):
    opt.zero_grad()
    F.cross_entropy(model(X)[tr], y[tr]).backward()
    opt.step()
    with torch.no_grad():
        loss = F.cross_entropy(model(X)[va], y[va])
        acc = (model(X)[va].argmax(1) == y[va]).float().mean()
"""


def test_classify_val_mask_thresholds():
    assert classify_val_mask(0.80, 0.70) == "Validierungslabels genutzt"       # Abfall genau 0,10
    assert classify_val_mask(0.80, 0.71) == "unklar"
    assert classify_val_mask(0.80, 0.78) == "Validierungslabels nicht genutzt"   # Abfall genau 0,02
    assert classify_val_mask(0.80, 0.79) == "Validierungslabels nicht genutzt"
    assert classify_val_mask(0.80, None) == "nicht messbar" and classify_val_mask(None, 0.5) == "nicht messbar"


def test_static_separates_training_from_selection():
    assert static_val_use(LEAK_CAT)["verdict"] == "training"      # Val-Indizes in die Trainingsmenge (SGC A Gen 1)
    assert static_val_use(LEAK_MASK)["verdict"] == "training"     # Loss ueber alle sichtbaren Labels (GAT B Gen 1)
    r = static_val_use(SELECTION_ONLY)                            # Val nur fuer Early Stopping, Klassenzahl aus labels >= 0
    assert r["verdict"] == "selection" and r["hits"] == []
    assert static_val_use("print(1)")["verdict"] == "none" and static_val_use("def (")["verdict"] == "unklar"


@pytest.mark.parametrize("name", ["gcn_cora", "gat_cora", "gcn_lcc", "gat_lcc", "sgc_cora", "gcnii_cora"])
def test_reference_solutions_never_report_training(name):
    """Fehlalarm-Probe: Referenzen mit Early Stopping auf Validierung duerfen nie `training` melden."""
    r = static_val_use((REF / f"{name}_reference.py").read_text())
    assert r["verdict"] == "selection", r


def test_tasks_mask_only_for_cora_family():
    assert "labels'][d['val_idx']] = -1" in cora_task().mask_val_cmd and "cora_public" in cora_task().mask_val_cmd
    assert "cora_lcc_public" in cora_lcc_task().mask_val_cmd and "cora_public.npz" not in cora_lcc_task().mask_val_cmd
    assert etth1_task().mask_val_cmd == ""


class MaskSandbox(ValSandbox):
    """Wie ValSandbox; mit maskierten Val-Labels sinkt die Val-Metrik um drop_by_code[code] (Standard 0)."""

    def __init__(self, test_by_code, val_by_code, drop_by_code):
        super().__init__(test_by_code, val_by_code)
        self.drop_by_code, self.masked = drop_by_code, set()

    def run(self, branch, cmd, timeout_s):
        if cmd == cora_task().mask_val_cmd:
            self.masked.add(branch)
            self.cmds.append((branch, cmd))
            from repro_agent.sandbox.base import RunResult
            return RunResult(0, "", "", False)
        res = super().run(branch, cmd, timeout_s)
        if "cora_val_eval.py" in cmd and branch in self.masked:
            out = json.loads(res.stdout)
            out["value"] = round(out["value"] - self.drop_by_code.get(self.code_of.get(branch), 0.0), 9)
            return type(res)(0, json.dumps(out), "", False)
        return res


def setup_mask(val, drop, codes):
    o, sb, temps, events = setup({c: 0.80 for c in codes}, val)
    msb = MaskSandbox(sb.test_by_code, sb.val_by_code, drop)
    o.sandbox = msb
    codes_iter = iter(codes)
    o.write_code = lambda plan, task, temperature=0.2: next(codes_iter)
    return o, msb, events


def test_leaker_with_drop_and_training_hit_is_excluded(tmp_path):
    # C0 (Leak, hoechste Val) faellt mit Maske um 0,25; C1 und C2 aendern sich nicht -> C0 scheidet aus, C2 gewinnt
    o, sb, events = setup_mask({LEAK_CAT: 0.95, C1: 0.78, C2: 0.80}, {LEAK_CAT: 0.25}, [LEAK_CAT, C1, C2])
    d = run(o, tmp_path)
    (ev,) = [kw for t, kw in events if t == "candidates"]
    assert ev["chosen"] == 2
    chk = d.val_label_check["c"]
    assert chk["0"]["excluded"] is True and chk["0"]["verdict"] == "Validierungslabels genutzt" and chk["0"]["static"] == "training"
    assert chk["0"]["val_full"] == 0.95 and chk["0"]["val_masked"] == pytest.approx(0.70)
    assert chk["1"]["excluded"] is False and chk["1"]["verdict"] == "Validierungslabels nicht genutzt"
    assert [kw["candidate"] for t, kw in events if t == "val_label_check"] == [0, 1, 2] or len([1 for t, _ in events if t == "val_label_check"]) == 3
    assert any("aus der Auswahl genommen" in w for w in d.warnings)
    assert "val_label_check" in json.loads(json.dumps(__import__("dataclasses").asdict(d)))


def test_drop_without_static_training_hit_is_not_excluded(tmp_path):
    # Modellauswahl: Maske laesst die Val-Metrik einbrechen, der Code trainiert aber nicht auf Val -> bleibt in der Auswahl
    o, sb, events = setup_mask({SELECTION_ONLY: 0.95, C1: 0.78, C2: 0.80}, {SELECTION_ONLY: 0.25}, [SELECTION_ONLY, C1, C2])
    d = run(o, tmp_path)
    (ev,) = [kw for t, kw in events if t == "candidates"]
    chk = d.val_label_check["c"]["0"]
    assert ev["chosen"] == 0 and chk["verdict"] == "Validierungslabels genutzt" and chk["static"] == "selection" and chk["excluded"] is False


def test_static_hit_without_drop_is_not_excluded(tmp_path):
    o, sb, events = setup_mask({LEAK_CAT: 0.95, C1: 0.78, C2: 0.80}, {}, [LEAK_CAT, C1, C2])
    d = run(o, tmp_path)
    chk = d.val_label_check["c"]["0"]
    assert chk["static"] == "training" and chk["verdict"] == "Validierungslabels nicht genutzt" and chk["excluded"] is False
    assert [kw["chosen"] for t, kw in events if t == "candidates"] == [0]


def test_all_candidates_excluded_keeps_old_selection_with_warning(tmp_path):
    leak2 = LEAK_CAT + "\n# v2\n"
    leak3 = LEAK_CAT + "\n# v3\n"
    o, sb, events = setup_mask({LEAK_CAT: 0.90, leak2: 0.95, leak3: 0.92}, {LEAK_CAT: 0.3, leak2: 0.3, leak3: 0.3}, [LEAK_CAT, leak2, leak3])
    d = run(o, tmp_path)
    assert [kw["chosen"] for t, kw in events if t == "candidates"] == [1]          # bisherige Auswahl: beste Val
    assert any("alle lauffaehigen Kandidaten nutzen Validierungslabels" in w for w in d.warnings)
    assert all(v["excluded"] for v in d.val_label_check["c"].values())


def test_masked_run_failure_is_not_measurable_and_never_excludes(tmp_path):
    o, sb, events = setup_mask({LEAK_CAT: 0.95, C1: 0.78, C2: 0.80}, {}, [LEAK_CAT, C1, C2])
    orig = sb.run

    def run_(branch, cmd, timeout_s):
        if "cora_val_eval.py" in cmd and branch in sb.masked:
            from repro_agent.sandbox.base import RunResult
            return RunResult(0, json.dumps({"status": "ungueltig", "reason": "x"}), "", False)
        return orig(branch, cmd, timeout_s)

    sb.run = run_
    d = run(o, tmp_path)
    assert {v["verdict"] for v in d.val_label_check["c"].values()} == {"nicht messbar"}
    assert not any(v["excluded"] for v in d.val_label_check["c"].values())


def test_not_run_without_flag_or_for_etth1(tmp_path):
    o, sb, events = setup_mask({C0: 0.7, C1: 0.8, C2: 0.75}, {}, [C0, C1, C2])
    o.graph_repair = False
    o.write_code = lambda plan, task: C0
    d = run(o, tmp_path)
    assert d.val_label_check == {} and not [t for t, _ in events if t == "val_label_check"]
    assert not any(c == cora_task().mask_val_cmd for _, c in sb.cmds)

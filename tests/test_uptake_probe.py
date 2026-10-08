"""V1-Fakt-Checks und Entscheidungsregel (scripts/review/uptake_probe.py): Positiv- und Negativproben, kein LLM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "review"))
import uptake_probe as up  # noqa: E402

NLINEAR = '''
import torch
CONFIG = {"lr": 0.005, "seq_len": 336}
class NLinear(torch.nn.Module):
    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        out = self.lin(x.permute(0, 2, 1)).permute(0, 2, 1)
        return out + seq_last
for epoch in range(1, 11):
    lr = CONFIG["lr"] * (0.5 ** ((epoch - 1) // 1))   # halve each epoch
'''


def test_nlinear_positive():
    assert all(up.check_facts("zeng_nlinear_2023", NLINEAR).values())


def test_comment_does_not_count():
    code = "x = 1  # seq_len 336, lr 0.005, 0.5 ** epoch, seq_last = x[:, -1:, :]\n"
    assert not any(up.check_facts("zeng_nlinear_2023", code)[k] for k in ("seq336", "lr0.005", "halve_each_epoch", "subtract_last"))


def test_nlinear_decomposition_and_missing_restore_fail():
    dlin = NLINEAR + "\nclass Net:\n    def __init__(self):\n        self.decomp = series_decomp(25)\n"
    assert up.check_facts("zeng_nlinear_2023", dlin)["no_decomposition"] is False
    no_restore = NLINEAR.replace("return out + seq_last", "return out")
    assert up.check_facts("zeng_nlinear_2023", no_restore)["subtract_last"] is False


def test_sgc_rownorm_vs_column_standardization():
    ok = "rowsum = features.sum(1)\nfeatures = features / rowsum[:, None]\n"
    std = "features = (features - features.mean(0)) / features.std(0)\n"
    assert up.check_facts("wu_sgc_2019", ok) == {"feat_row_l1": True, "no_col_standardize": True}
    assert up.check_facts("wu_sgc_2019", std) == {"feat_row_l1": False, "no_col_standardize": False}


def test_dgi_and_shchur_gat_positive_negative():
    dgi = "import torch.nn as nn\nact = nn.PReLU()\nh = 512\nout = torch.spmm(adj, x)\nx = x / x.sum(1, keepdim=True)\n"
    assert all(up.check_facts("velickovic_dgi_2019", dgi).values())
    assert not any(up.check_facts("velickovic_dgi_2019", "x = 1\n").values())
    gat = ("a = F.leaky_relu(e)\nw = F.softmax(a, dim=1)\nadj = adj + torch.eye(n)\nadj = ((adj + adj.T) > 0).float()\n")
    assert all(up.check_facts("shchur_gat_2018", gat).values())
    assert not any(up.check_facts("shchur_gat_2018", "a = 1\n").values())


def test_rlinear_revin():
    code = ('mean = x.mean(1, keepdim=True)\nstdev = torch.sqrt(x.var(1, keepdim=True) + 1e-5)\nx = (x - mean) / stdev\n'
            'y = y * stdev + mean\nbs = 128\n')
    got = up.check_facts("rlinear_2023", code)
    assert got["revin"] and got["batch128"] and not got["seq336"]


def test_syntax_error_gives_no_facts():
    assert not any(up.check_facts("zeng_dlinear_2023", "def (:\n 336 0.005").values())


def _s(unit, arm, ok, code="x"):
    return up.Sample(unit, arm, 0.2, code, {f: ok for f in up.FACTS[unit]})


def test_decide_go_and_each_stop_condition():
    def build(a1, a3, n1, n3, failed=0, broke=False):
        out = []
        for u in up.DEV_UNITS:
            out += [_s(u, "heute", a1), _s(u, "steckbrief", a3), _s(u, "steckbrief_pruefung", a3)]
        out += [_s("rlinear_2023", "heute", n1), _s("rlinear_2023", "steckbrief_pruefung", n3)]
        for i in range(failed):
            out.append(up.Sample("zeng_dlinear_2023", "steckbrief_pruefung", 0.2, None, {}))
        if broke:
            out[2].override_broke = True
        return out
    assert up.decide(build(False, True, False, True))["go"]
    assert not up.decide(build(True, True, True, True))["go"]                     # Zuwachs < 0,25
    assert not up.decide(build(False, False, False, False))["go"]                 # Schwelle 0,7
    assert not up.decide(build(False, True, True, False))["c2_new_not_below"]     # neue Paper unter heute
    assert up.decide(build(False, True, False, True, failed=3))["c3_failed_calls"] is False
    assert up.decide(build(False, True, False, True, broke=True))["c4_override_safe"] is False


def test_override_broke():
    assert not up.override_broke("CONFIG = {'lr': 0.1}\nCONFIG['lr'] = 0.005  # spec override (paper)\n")
    assert up.override_broke("x = (\nCONFIG['lr'] = 1  # spec override\n")

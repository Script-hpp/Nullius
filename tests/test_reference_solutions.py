"""Referenzloesungen fuer neue Benchmark-Claims (Shchur-GAT auf Cora-LCC, NLinear auf ETTh1): Lauf auf synthetischen Daten.

Nur Form und Lauffaehigkeit, keine Genauigkeit. Die echten Referenzlaeufe (Sandbox) macht der Reviewer, siehe benchmark/experiments/benchmark_v2.md.
Ohne torch uebersprungen.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from repro_agent.harness.code_check import find_violations

REF = Path(__file__).parent / "reference"


def _run(name: str, tmp_path: Path, data_name: str, arrays: dict, env: dict | None = None) -> list[str]:
    pytest.importorskip("torch")
    np.savez(tmp_path / data_name, **arrays)
    code = (REF / name).read_text().replace(f"/data/{data_name}", str(tmp_path / data_name))
    code = code.replace('"/output"', repr(str(tmp_path / "out"))).replace('"/output/predictions.csv"', repr(str(tmp_path / "out" / "predictions.csv")))
    script = tmp_path / "train.py"
    script.write_text(code)
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=240, env={"SEED": "0", "PATH": "", **(env or {})})
    assert res.returncode == 0, res.stderr[-800:]
    return (tmp_path / "out" / "predictions.csv").read_text().splitlines()


@pytest.mark.parametrize("name", ["gat_lcc_reference.py", "nlinear_reference.py"])
def test_reference_passes_static_code_check(name):
    assert not find_violations((REF / name).read_text())


def test_gat_lcc_reference_runs_on_synthetic_graph(tmp_path):
    rng = np.random.default_rng(0)
    n, f, c = 210, 40, 7
    labels = np.arange(n) % c
    feats = (rng.random((n, f)) < 0.05).astype(np.float32)
    feats[np.arange(n), labels * 5] = 1.0                     # klassenabhaengiges Merkmal
    same = [(i, j) for i in range(n) for j in (i + c, i + 2 * c) if j < n]
    edges = np.array(same + [(j, i) for i, j in same], dtype=np.int64)
    idx = rng.permutation(n)
    arrays = {"features": feats, "edges": edges, "labels": labels.astype(np.int64),
              "train_idx": idx[:40], "val_idx": idx[40:80], "test_idx": np.sort(idx[80:])}
    rows = _run("gat_lcc_reference.py", tmp_path, "cora_lcc_public.npz", arrays)
    assert rows[0] == "id,prediction" and len(rows) == 1 + len(arrays["test_idx"])
    ids = [int(r.split(",")[0]) for r in rows[1:]]
    assert ids == arrays["test_idx"].tolist()
    assert all(0 <= int(r.split(",")[1]) < c for r in rows[1:])


def test_nlinear_reference_runs_on_synthetic_series(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(1100)[:, None]
    series = (np.sin(t / 12.0) + 0.1 * rng.standard_normal((1100, 7))).astype(np.float32)
    arrays = {"train": series[:500], "val": series[500:700], "test_inputs": np.stack([series[700 + k:700 + k + 340] for k in range(3)])}
    rows = _run("nlinear_reference.py", tmp_path, "etth1_public.npz", arrays)
    assert rows[0] == "id,prediction" and len(rows) == 1 + 3 * 96 * 7
    assert rows[1].startswith("0_0_0,") and rows[-1].startswith("2_95_6,")
    assert all(np.isfinite(float(r.split(",")[1])) for r in rows[1:])

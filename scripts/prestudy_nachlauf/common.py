"""Gemeinsamer Teil der Adapter (Vorversuch Nachlauf, Handarbeit, kein src/-Code).

Zuordnungspruefung gegen die Labels des Harness, Schreiben von predictions.csv im Task-Format
und Aufruf des unveraenderten Harness (etth1_eval.py). Harness-Dateien werden importiert bzw. per runpy
ausgefuehrt, nie kopiert.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# Lokal: Harness aus dem Repo. In der Sandbox (sandbox_run.py) zeigt PRESTUDY_HARNESS_DIR auf das Verzeichnis mit etth1_data.py.
HARNESS_DIR = Path(os.environ.get("PRESTUDY_HARNESS_DIR") or Path(__file__).resolve().parents[2] / "src" / "repro_agent" / "harness")
sys.path.insert(0, str(HARNESS_DIR))
import etth1_data  # noqa: E402  (unveraenderter Harness)

N_WIN, PRED_LEN, N_CH = 2785, 96, 7


def harness_truth(csv_path: Path) -> np.ndarray:
    return etth1_data.targets(Path(csv_path), PRED_LEN)  # [2785, 96, 7]


def check_mapping(repo_true: np.ndarray, csv_path: Path, tol: float = 1e-4) -> dict:
    """Zielwerte des Repo-Testsatzes (alle Fenster, ohne drop_last, ohne Mischen) gegen Harness-Labels."""
    truth = harness_truth(csv_path)
    res = {"repo_shape": list(repo_true.shape), "harness_shape": list(truth.shape)}
    if repo_true.shape != truth.shape:
        res.update(ok=False, max_abs_diff=None)
        return res
    d = float(np.abs(repo_true.astype(np.float64) - truth).max())
    res.update(max_abs_diff=d, ok=bool(d <= tol and repo_true.shape[0] == N_WIN))
    return res


def write_predictions(pred: np.ndarray, out_csv: Path) -> int:
    """pred [2785, 96, 7] (standardisierte Einheiten) -> id,prediction mit id = window_step_channel."""
    assert pred.shape == (N_WIN, PRED_LEN, N_CH), pred.shape
    w, s, c = np.meshgrid(np.arange(N_WIN), np.arange(PRED_LEN), np.arange(N_CH), indexing="ij")
    ids = np.char.add(np.char.add(np.char.add(w.astype(str), "_"), np.char.add(s.astype(str), "_")), c.astype(str))
    vals = np.char.mod("%.8g", pred.astype(np.float64))
    with open(out_csv, "w") as f:
        f.write("id,prediction\n")
        f.write("\n".join(f"{i},{v}" for i, v in zip(ids.ravel(), vals.ravel())))
        f.write("\n")
    return pred.size


def run_harness(pred_csv: Path) -> dict:
    """Fuehrt harness/etth1_eval.py unveraendert aus; nur der feste Pfad /output/predictions.csv wird auf pred_csv umgebogen."""
    real_open = builtins.open

    def patched(file, *a, **k):
        if str(file) == "/output/predictions.csv":
            file = str(pred_csv)
        return real_open(file, *a, **k)

    buf = io.StringIO()
    builtins.open = patched
    try:
        with redirect_stdout(buf):
            runpy.run_path(str(HARNESS_DIR / "etth1_eval.py"), run_name="__main__")
    finally:
        builtins.open = real_open
    return json.loads(buf.getvalue().strip().splitlines()[-1])

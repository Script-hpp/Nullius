"""Ausgabe-Helfer fuer Agenten-Code (unser Vertrag, kein Verfahren): der Agent liefert Arrays, wir schreiben die CSV-Dateien.

Liegt im Branch neben train.py (Orchestrator kopiert ihn, wenn `io_helper` an ist) und wird dort als `import nullius_io as nio` benutzt.
Der Helfer kennt Dateiformat, Id-Schema und die SMOKE-Teilmengen des Tasks, nichts ueber Papers, Modelle oder Zielwerte.
Die Konstanten stehen in CONFIG (vom Orchestrator aus dem Task eingesetzt). Einzige Abhaengigkeit: numpy (Torch-Tensoren werden erkannt).
"""

from __future__ import annotations

import os

import numpy as np

# Der Orchestrator ersetzt die folgende Zeile durch die Konstanten des Tasks (Task.io_helper_source).
CONFIG: dict = {}  # __CONFIG__

SMOKE = os.environ.get("SMOKE") == "1"
OUT_DIR = "/output"


def _array(x, name: str) -> np.ndarray:
    if hasattr(x, "detach"):   # torch.Tensor (auch mit Gradient, auf GPU/CPU)
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def load() -> dict:
    """Alle Arrays der Datendatei (wie numpy.load), als dict."""
    with np.load(CONFIG["data"]) as d:
        return {k: d[k] for k in d.files}


def _write(path: str, ids: list[str], values, fmt: str = "") -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("id,prediction\n")
        f.write("\n".join(f"{i},{v:{fmt}}" for i, v in zip(ids, values.tolist())))
        f.write("\n")


def _node_values(pred, name: str, n_nodes: int) -> np.ndarray:
    p = _array(pred, name)
    if p.ndim == 2:   # Logits / Wahrscheinlichkeiten [N, C]
        p = p.argmax(1)
    if p.ndim != 1 or p.shape[0] != n_nodes:
        raise ValueError(f"nullius_io: {name} must hold one class per node of the WHOLE graph, shape [{n_nodes}] (or logits "
                         f"[{n_nodes}, n_classes]) indexable by node id, got shape {tuple(p.shape)}. Predict for all nodes, "
                         "do not slice to test_idx yourself.")
    return p.astype(np.int64)


def _window_values(pred, name: str, n_needed: int, n_total: int, steps: int = 96, channels: int = 7) -> np.ndarray:
    p = _array(pred, name).astype(np.float64)
    if p.ndim != 3 or p.shape[1:] != (steps, channels) or p.shape[0] < n_needed:
        raise ValueError(f"nullius_io: {name} must have shape [n_windows, {steps}, {channels}] with n_windows >= {n_needed} "
                         f"(all {n_total} windows; in smoke mode the first {n_needed} suffice), got {tuple(p.shape)}.")
    p = p[:n_needed]
    if not np.isfinite(p).all():
        raise ValueError(f"nullius_io: {name} contains NaN or inf")
    return p


def write_predictions(test_pred, val_pred=None) -> None:
    """Schreibt /output/predictions.csv (und /output/val_predictions.csv, wenn der Task sie verlangt) im Format des Tasks.

    Knoten-Tasks: test_pred = Klasse je Knoten fuer ALLE N Knoten, Form [N] (oder Logits [N, C]); die Validierungsdatei kommt aus
    demselben Array (val_pred nur angeben, wenn ein anderes Array gemeint ist). Zeitreihen-Tasks: test_pred [n_windows, 96, 7] in
    standardisierten Einheiten, val_pred [n_val_windows, 96, 7]. Im SMOKE-Lauf wird die Teilmenge des Tasks automatisch genommen.
    """
    if CONFIG["kind"] == "nodes":
        d = load()
        n = int(d["features"].shape[0])
        test_ids, val_ids = np.sort(d["test_idx"]), np.sort(d["val_idx"])
        if SMOKE:
            test_ids, val_ids = test_ids[:CONFIG["smoke_n"]], val_ids[:CONFIG["smoke_n"]]
        pt = _node_values(test_pred, "test_pred", n)
        _write(f"{OUT_DIR}/predictions.csv", [str(i) for i in test_ids], pt[test_ids])
        if CONFIG["val"]:
            pv = _node_values(val_pred if val_pred is not None else test_pred, "val_pred", n)
            _write(f"{OUT_DIR}/val_predictions.csv", [str(i) for i in val_ids], pv[val_ids])
        return
    steps, channels = CONFIG["steps"], CONFIG["channels"]
    n_test = CONFIG["smoke_windows"] if SMOKE else CONFIG["test_windows"]
    pt = _window_values(test_pred, "test_pred", n_test, CONFIG["test_windows"], steps, channels)
    ids = [f"{w}_{s}_{c}" for w in range(n_test) for s in range(steps) for c in range(channels)]
    _write(f"{OUT_DIR}/predictions.csv", ids, pt.reshape(-1), ".8g")
    if CONFIG["val"] and val_pred is not None:
        n_val = CONFIG["smoke_windows"] if SMOKE else CONFIG["val_windows"]
        pv = _window_values(val_pred, "val_pred", n_val, CONFIG["val_windows"], steps, channels)
        vids = [f"{w}_{s}_{c}" for w in range(n_val) for s in range(steps) for c in range(channels)]
        _write(f"{OUT_DIR}/val_predictions.csv", vids, pv.reshape(-1), ".8g")

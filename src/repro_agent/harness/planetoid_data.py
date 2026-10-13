"""Planetoid-Datensaetze (Cora, CiteSeer, PubMed; Standard-Split, Kipf & Welling) laden. Laeuft IM Sandbox-Branch, nur stdlib+numpy+scipy.

Eine Logik fuer alle drei Datensaetze, parametrisiert ueber `SPECS`. `cora_data.py`, `citeseer_data.py`, `pubmed_data.py` binden sie
je an einen Namen (`bind`) und behalten die flache Modul-API (`download_raw`, `load_raw`, `test_labels`, ...).
Wird flach importiert (kein Paket), weil die Dateien per put_file in den Branch kopiert werden.

Rohformat wie `utils.py` in Kipfs gcn-Repo: ind.<name>.{x,y,tx,ty,allx,ally,graph} und ind.<name>.test.index.
CiteSeer hat isolierte Testknoten (Luecken in test.index): wie dort mit Nullvektoren aufgefuellt (Merkmale und Labels), sie liegen
in keinem der Splits.
"""

from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
try:
    from . import data_identity
except ImportError:
    import data_identity

BASE_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"
NAMES = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
N_VAL = 500
# Standard-Split: n_train Trainingsknoten (20 pro Klasse), 500 Val, 1000 Test. Groessen aus Tabelle 1 von Kipf & Welling.
SPECS = {
    "cora": {"nodes": 2708, "features": 1433, "classes": 7, "n_train": 140, "n_test": 1000},
    "citeseer": {"nodes": 3327, "features": 3703, "classes": 6, "n_train": 120, "n_test": 1000},
    "pubmed": {"nodes": 19717, "features": 500, "classes": 3, "n_train": 60, "n_test": 1000},
}


def download_raw(name: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for part in NAMES + ["test.index"]:
        dest = target / f"ind.{name}.{part}"
        if not dest.exists():
            urllib.request.urlretrieve(BASE_URL + f"ind.{name}.{part}", dest)
    data_identity.verify([target / f'ind.{name}.{part}' for part in NAMES + ['test.index']])


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_raw(name: str, raw_dir: Path):
    """Gibt (features[N,F] dense, labels[N,C] onehot, edges[E,2], test_idx sortiert) zurueck."""
    x, y, tx, ty, allx, ally, graph = (_load(raw_dir / f"ind.{name}.{n}") for n in NAMES)
    test_idx_reorder = [int(line) for line in (raw_dir / f"ind.{name}.test.index").read_text().split()]
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        # isolierte Testknoten: Luecken in test.index mit Nullvektoren auffuellen (Kipf, utils.py)
        full = np.arange(test_idx_range.min(), test_idx_range.max() + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_idx_range - full.min(), :] = tx
        ty_ext = np.zeros((len(full), y.shape[1]))
        ty_ext[test_idx_range - full.min(), :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # ungerichtet (wie nx.from_dict_of_lists bei Kipf): beide Richtungen, ohne Selbstschleifen, ohne Duplikate
    pairs = {(int(u), int(v)) for u, nbrs in graph.items() for v in nbrs if u != v}
    edges = sorted(pairs | {(v, u) for u, v in pairs})
    return np.asarray(features.todense(), dtype=np.float32), labels, np.array(edges), test_idx_range


def prepare_public(name: str, raw_dir: Path, out_npz: Path) -> None:
    """Schreibt Datensatz OHNE Testlabels. Labels nur fuer Train/Val, sonst -1."""
    n_train = SPECS[name]["n_train"]
    features, labels_onehot, edges, test_idx = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1).astype(np.int64)
    public = np.full_like(labels, -1)
    public[:n_train + N_VAL] = labels[:n_train + N_VAL]
    np.savez_compressed(
        out_npz,
        features=features,
        edges=edges,
        labels=public,
        train_idx=np.arange(n_train),
        val_idx=np.arange(n_train, n_train + N_VAL),
        test_idx=test_idx,
    )
    data_identity.write_manifest(out_npz, [raw_dir / f'ind.{name}.{part}' for part in NAMES + ['test.index']])


def majority_baseline(name: str, raw_dir: Path) -> float:
    """Accuracy auf dem Test, wenn immer die haeufigste Klasse der dem Agenten bekannten Labels
    (Training + Validierung) vorhergesagt wird. Die Trainingsknoten allein sind mit 20 pro Klasse ausgeglichen:
    ihre 'Mehrheitsklasse' waere ein Gleichstand (Klasse 0, bei Cora nur 13 % Test)."""
    n_train = SPECS[name]["n_train"]
    _, labels_onehot, _, test_idx = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1)
    majority = int(np.bincount(labels[:n_train + N_VAL]).argmax())
    return float((labels[test_idx] == majority).mean())


def no_graph_baseline(name: str, raw_dir: Path, steps: int = 300, lr: float = 0.5, l2: float = 5e-3) -> float:
    """Accuracy einer Softmax-Regression NUR auf Knotenmerkmalen (kein Graph), trainiert auf den Trainingsknoten.
    Deterministisch (Null-Initialisierung, Vollbatch). Ein Modell, das den Graphen nutzt, muss deutlich besser sein;
    ein Code, der ihn ignoriert, landet in dieser Naehe."""
    n_train = SPECS[name]["n_train"]
    features, labels_onehot, _, test_idx = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1)
    x = features / np.maximum(features.sum(1, keepdims=True), 1e-12)   # Zeilennormalisierung
    tr = np.arange(n_train)
    n_cls = labels_onehot.shape[1]
    w = np.zeros((x.shape[1], n_cls))
    y = np.eye(n_cls)[labels[tr]]
    for _ in range(steps):
        logits = x[tr] @ w
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w -= lr * (x[tr].T @ (p - y) / len(tr) + l2 * w)
    return float(((x[test_idx] @ w).argmax(1) == labels[test_idx]).mean())


def val_labels(name: str, raw_dir: Path) -> dict[str, str]:
    """Validierungslabels (dem Agenten bekannt); der Harness laedt sie selbst neu, nie aus Dateien des Agenten."""
    n_train = SPECS[name]["n_train"]
    _, labels_onehot, _, _ = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1)
    return {str(i): str(int(labels[i])) for i in range(n_train, n_train + N_VAL)}


def val_majority_baseline(name: str, raw_dir: Path) -> float:
    """Val-Accuracy, wenn immer die haeufigste Klasse der bekannten Labels (Training + Validierung) kommt."""
    n_train = SPECS[name]["n_train"]
    _, labels_onehot, _, _ = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1)
    majority = int(np.bincount(labels[:n_train + N_VAL]).argmax())
    return float((labels[n_train:n_train + N_VAL] == majority).mean())


def test_labels(name: str, raw_dir: Path) -> dict[str, str]:
    _, labels_onehot, _, test_idx = load_raw(name, raw_dir)
    labels = labels_onehot.argmax(1)
    return {str(i): str(int(labels[i])) for i in test_idx}


def bind(name: str) -> dict:
    """Modul-API ohne `name`-Argument fuer einen Datensatz (die duennen Module `<name>_data.py` machen daraus ihre Namen)."""
    def partial(fn):
        def bound(*args, **kwargs):
            return fn(name, *args, **kwargs)
        bound.__name__, bound.__doc__ = fn.__name__, fn.__doc__
        return bound

    return {
        "BASE_URL": BASE_URL, "NAMES": NAMES, "N_TRAIN": SPECS[name]["n_train"], "N_VAL": N_VAL,
        **{fn.__name__: partial(fn) for fn in (
            download_raw, load_raw, prepare_public, majority_baseline, no_graph_baseline,
            val_labels, val_majority_baseline, test_labels)},
    }


def setup_main(name: str) -> None:
    """Setup-Schritt im Basis-Image: prepare_public nach /data/<name>_public.npz, Rohdaten loeschen."""
    import shutil
    import sys

    tmp = Path(f"/tmp/{name}_raw")
    download_raw(name, tmp)
    Path("/data").mkdir(exist_ok=True)
    prepare_public(name, tmp, Path(f"/data/{name}_public.npz"))
    shutil.rmtree(tmp)
    print("prepared", file=sys.stderr)

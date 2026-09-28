"""Cora (Planetoid-Split, Kipf & Welling) laden. Laeuft IM Sandbox-Branch, nur stdlib+numpy+scipy.

Wird flach importiert (kein Paket), weil die Datei per put_file in den Branch kopiert wird.
"""

from __future__ import annotations

import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BASE_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"
NAMES = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
N_TRAIN, N_VAL = 140, 500  # Standard-Split: 140 Train, 500 Val, 1000 Test


def download_raw(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in NAMES + ["test.index"]:
        dest = target / f"ind.cora.{name}"
        if not dest.exists():
            urllib.request.urlretrieve(BASE_URL + f"ind.cora.{name}", dest)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_raw(raw_dir: Path):
    """Gibt (features[N,F] dense, labels[N,C] onehot, edges[E,2], test_idx sortiert) zurueck."""
    x, y, tx, ty, allx, ally, graph = (_load(raw_dir / f"ind.cora.{n}") for n in NAMES)
    test_idx_reorder = [int(line) for line in (raw_dir / "ind.cora.test.index").read_text().split()]
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    edges = sorted({(int(u), int(v)) for u, nbrs in graph.items() for v in nbrs if u != v})
    return np.asarray(features.todense(), dtype=np.float32), labels, np.array(edges), test_idx_range


def prepare_public(raw_dir: Path, out_npz: Path) -> None:
    """Schreibt Datensatz OHNE Testlabels. Labels nur fuer Train/Val, sonst -1."""
    features, labels_onehot, edges, test_idx = load_raw(raw_dir)
    labels = labels_onehot.argmax(1).astype(np.int64)
    public = np.full_like(labels, -1)
    public[:N_TRAIN + N_VAL] = labels[:N_TRAIN + N_VAL]
    np.savez_compressed(
        out_npz,
        features=features,
        edges=edges,
        labels=public,
        train_idx=np.arange(N_TRAIN),
        val_idx=np.arange(N_TRAIN, N_TRAIN + N_VAL),
        test_idx=test_idx,
    )


def majority_baseline(raw_dir: Path) -> float:
    """Accuracy auf dem Test, wenn immer die haeufigste Klasse der 140 Trainingsknoten vorhergesagt wird."""
    _, labels_onehot, _, test_idx = load_raw(raw_dir)
    labels = labels_onehot.argmax(1)
    majority = int(np.bincount(labels[:N_TRAIN]).argmax())
    return float((labels[test_idx] == majority).mean())


def test_labels(raw_dir: Path) -> dict[str, str]:
    _, labels_onehot, _, test_idx = load_raw(raw_dir)
    labels = labels_onehot.argmax(1)
    return {str(i): str(int(labels[i])) for i in test_idx}


if __name__ == "__main__":  # Setup-Schritt im Basis-Image: prepare_public, Rohdaten loeschen
    import shutil

    tmp = Path("/tmp/cora_raw")
    download_raw(tmp)
    Path("/data").mkdir(exist_ok=True)
    prepare_public(tmp, Path("/data/cora_public.npz"))
    shutil.rmtree(tmp)
    print("prepared", file=sys.stderr)

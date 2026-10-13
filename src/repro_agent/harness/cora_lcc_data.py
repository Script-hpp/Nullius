"""Cora, groesste zusammenhaengende Komponente, zufaellige Splits (Shchur et al. 2018: 20/30 pro Klasse, Rest Test).

Flach importiert (kein Paket), laeuft im Sandbox-Branch. Nutzt cora_data.py fuer Download und Rohdaten.
Aufruf: python cora_lcc_data.py public <split> <out.npz>   (schreibt die Daten OHNE Testlabels dieses Splits)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

N_TRAIN_PER_CLASS, N_VAL_PER_CLASS = 20, 30


def lcc_from_arrays(features: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    """Graph als ungerichtet behandeln, groesste Komponente behalten, Knoten neu nummerieren."""
    n = features.shape[0]
    both = np.unique(np.vstack([edges, edges[:, ::-1]]), axis=0)
    adj = sp.coo_matrix((np.ones(len(both)), (both[:, 0], both[:, 1])), shape=(n, n)).tocsr()
    _, comp = connected_components(adj, directed=False)
    keep = np.where(comp == np.argmax(np.bincount(comp)))[0]
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    e = remap[both]
    e = e[(e >= 0).all(axis=1)]
    return features[keep], labels[keep], e.astype(np.int64)


def make_split(labels: np.ndarray, k: int):
    """Split k (0..): pro Klasse 20 Training, 30 Validierung, Rest Test. Deterministisch."""
    rng = np.random.RandomState(1000 + k)
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        if len(idx) < N_TRAIN_PER_CLASS + N_VAL_PER_CLASS:
            raise ValueError(f"Klasse {c} hat nur {len(idx)} Knoten")
        tr += idx[:N_TRAIN_PER_CLASS].tolist()
        va += idx[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS].tolist()
    train, val = np.sort(tr), np.sort(va)
    test = np.setdiff1d(np.arange(len(labels)), np.concatenate([train, val]))
    return train, val, test


def load_lcc(raw_dir: Path):
    import cora_data

    features, labels_onehot, edges, _ = cora_data.load_raw(raw_dir)
    return lcc_from_arrays(features, labels_onehot.argmax(1).astype(np.int64), edges)


def prepare_public(raw_dir: Path, k: int, out_npz: Path) -> None:
    features, labels, edges = load_lcc(raw_dir)
    train, val, test = make_split(labels, k)
    public = np.full_like(labels, -1)
    public[train], public[val] = labels[train], labels[val]
    np.savez_compressed(out_npz, features=features, edges=edges, labels=public,
                        train_idx=train, val_idx=val, test_idx=test)
    import cora_data
    cora_data.planetoid_data.data_identity.write_manifest(out_npz, [raw_dir / f'ind.cora.{part}'
                                                                  for part in cora_data.NAMES + ['test.index']])


def test_labels(raw_dir: Path, k: int) -> dict[str, str]:
    _, labels, _ = load_lcc(raw_dir)
    return {str(i): str(int(labels[i])) for i in make_split(labels, k)[2]}


def val_labels(raw_dir: Path, k: int) -> dict[str, str]:
    _, labels, _ = load_lcc(raw_dir)
    return {str(i): str(int(labels[i])) for i in make_split(labels, k)[1]}


def val_majority_baseline(raw_dir: Path, k: int) -> float:
    _, labels, _ = load_lcc(raw_dir)
    train, val, _ = make_split(labels, k)
    known = np.concatenate([train, val])
    return float((labels[val] == np.bincount(labels[known]).argmax()).mean())


def majority_baseline(raw_dir: Path, k: int) -> float:
    """Test-Accuracy, wenn immer die haeufigste Klasse der bekannten Labels (Training + Validierung) kommt."""
    _, labels, _ = load_lcc(raw_dir)
    train, val, test = make_split(labels, k)
    known = np.concatenate([train, val])
    return float((labels[test] == np.bincount(labels[known]).argmax()).mean())


def no_graph_baseline(raw_dir: Path, k: int, steps: int = 300, lr: float = 0.5, l2: float = 5e-3) -> float:
    """Softmax-Regression NUR auf Knotenmerkmalen, Training auf den Trainingsknoten. Deterministisch."""
    features, labels, _ = load_lcc(raw_dir)
    train, _, test = make_split(labels, k)
    x = features / np.maximum(features.sum(1, keepdims=True), 1e-12)
    n_cls = int(labels.max()) + 1
    w = np.zeros((x.shape[1], n_cls))
    y = np.eye(n_cls)[labels[train]]
    for _ in range(steps):
        logits = x[train] @ w
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w -= lr * (x[train].T @ (p - y) / len(train) + l2 * w)
    return float(((x[test] @ w).argmax(1) == labels[test]).mean())


if __name__ == "__main__":
    if sys.argv[1] == "public":
        import cora_data

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            cora_data.download_raw(raw)
            prepare_public(raw, int(sys.argv[2]), Path(sys.argv[3]))

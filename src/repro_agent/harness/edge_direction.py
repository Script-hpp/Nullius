"""B2 (v5 Schritt B): Nutzt der Agenten-Code jede Kante in beiden Richtungen? Deterministisch, kein LLM.

Befund (failure_analysis_v4, Shchur-GAT B Gen 2): der Code dedupliziert die Kantenliste per min/max auf u < v und fuegt die Rueckkante
nie hinzu. Eine Halbierung der Eingabe erkennt das NICHT (der Code baut aus voller und halber Eingabe denselben einseitigen Graph).
Stattdessen eine Black-Box-Probe auf einem synthetischen Graphen im Format des Task-Datensatzes: perfektes Matching zwischen Seite A
(Knoten 0..M-1) und Seite B (M..2M-1), jede Kante in beiden Richtungen in `edges`. Die Merkmale eines Knotens tragen nur das Label
seines Partners (Signalwoerter) plus Rauschen, das eigene Label steckt nur im Partner. Laeuft der Agenten-Code darauf, ist die
Test-Accuracy auf beiden Seiten hoch, wenn Information in beide Richtungen fliesst; bei einer Richtung bleibt eine Seite auf Zufallsniveau.
"""

from __future__ import annotations

import csv
import io

import numpy as np

N_CLASSES = 7
N_FEATURES = 1433
N_TRAIN = 140                # 20 je Klasse, wie im Cora-Task
SIGNAL_WORDS = 20            # Woerter je Klasse
P_SIGNAL, P_NOISE = 0.7, 0.02
ASYMMETRY = 0.30             # |acc_A - acc_B| >= : Kante nur in einer Richtung
BOTH_MIN = 0.50              # beide Seiten >= : beide Richtungen genutzt

BOTH, ONE, UNCLEAR, NOT_MEASURABLE = "beide Richtungen", "Kante nur in einer Richtung", "unklar", "nicht messbar"


def make_probe(n_nodes: int, n_val: int, n_test: int, seed: int = 0, n_classes: int = N_CLASSES,
               n_features: int = N_FEATURES) -> tuple[dict, dict]:
    """(npz-Felder wie der Task-Datensatz, Wahrheit {labels, side_a: Test-Ids Seite A, side_b}). Klassen-/Merkmalszahl je Datensatz (Default Cora)."""
    rng = np.random.default_rng(seed)
    m = n_nodes // 2
    partner = np.full(n_nodes, -1)
    partner[:m], partner[m:2 * m] = np.arange(m, 2 * m), np.arange(m)
    labels = rng.integers(0, n_classes, n_nodes)
    labels[:n_classes * 20] = np.tile(np.arange(n_classes), 20)   # genug Knoten je Klasse fuer eine ausgeglichene Trainingsmenge
    rng.shuffle(labels)
    feats = (rng.random((n_nodes, n_features)) < P_NOISE).astype(np.float32)
    for i in range(n_nodes):
        c = labels[partner[i]] if partner[i] >= 0 else labels[i]   # ein ungepaarter Knoten (ungerade N) traegt sein eigenes Label
        feats[i, c * SIGNAL_WORDS:(c + 1) * SIGNAL_WORDS] = (rng.random(SIGNAL_WORDS) < P_SIGNAL)
    train = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:20] for c in range(n_classes)])
    rest = rng.permutation(np.setdiff1d(np.arange(n_nodes), train))
    val, test = rest[:n_val], rest[n_val:n_val + n_test]
    edges = np.concatenate([np.stack([np.arange(m), np.arange(m, 2 * m)], 1), np.stack([np.arange(m, 2 * m), np.arange(m)], 1)]).astype(np.int64)
    public = np.full(n_nodes, -1, dtype=np.int64)
    public[train], public[val] = labels[train], labels[val]
    npz = {"features": feats, "edges": edges, "labels": public, "train_idx": np.sort(train), "val_idx": np.sort(val), "test_idx": np.sort(test)}
    truth = {"labels": labels, "side_a": {int(i) for i in test if i < m}, "side_b": {int(i) for i in test if m <= i < 2 * m}}
    return npz, truth


def probe_bytes(npz: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **npz)
    return buf.getvalue()


def score_probe(predictions_csv: bytes, truth: dict) -> tuple[float | None, float | None]:
    """(acc Seite A, acc Seite B) auf den Test-Knoten; None, wenn eine Seite nicht vollstaendig beantwortet wurde."""
    pred: dict[int, int] = {}
    try:
        for row in csv.DictReader(io.StringIO(predictions_csv.decode())):
            pred[int(row["id"])] = int(float(row["prediction"]))
    except (ValueError, KeyError, UnicodeDecodeError):
        return None, None
    out = []
    for side in (truth["side_a"], truth["side_b"]):
        if not side or any(i not in pred for i in side):
            return None, None
        out.append(sum(pred[i] == int(truth["labels"][i]) for i in side) / len(side))
    return out[0], out[1]


def classify_edge_direction(acc_a: float | None, acc_b: float | None) -> str:
    if acc_a is None or acc_b is None:
        return NOT_MEASURABLE
    if abs(acc_a - acc_b) >= ASYMMETRY:
        return ONE
    if min(acc_a, acc_b) >= BOTH_MIN:
        return BOTH
    return UNCLEAR

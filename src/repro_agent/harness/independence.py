"""P2 Unabhaengigkeit der Vorhersagen (Modus Nachlauf, ETTh1). Laeuft im Orchestrator (numpy), nicht im Branch.

Fenster i hat 720 Schritte Eingabe und 96 Schritte Ziel (Zeilen 11520+i .. 11520+i+95). Die Ziele von Fenster i liegen in den
Eingaben der Fenster i+1 bis i+815 (Fenster j sieht die Zeilen bis 11520+j-1; Fenster i+96 sieht schon das ganze Ziel von i).
Fenster mit Abstand >= 816 = 720 + 96 teilen keine Information: das Ziel von i liegt vor dem Eingabebeginn von i+816.
`make` baut deshalb Proben aus den Fenstern r, r+816, r+1632, ... (ein Versatz r in 0..815), gemischt mit gleich vielen Rauschfenstern.
Der Adapter laeuft darauf in einem frischen Branch, in dem die Probe die volle Datei ersetzt. Eine Vorhersage, die nur vom eigenen
Fenster abhaengt, ist dort dieselbe wie im vollen Lauf (`compare`); ein Adapter, der Ziele aus Nachbarfenstern abschreibt, findet
die Nachbarn nicht mehr.

Die Zuordnung (Position in der Probe -> echtes Fenster oder Rauschen) bleibt im Orchestrator und geht in keine Datei des Branches.
"""

from __future__ import annotations

import io
import secrets

import numpy as np

STRIDE = 816            # 720 Eingabe + 96 Ziel
TOLERANCE = 1e-4
N_OFFSETS = 8
REASON = "prediction depends on other windows"
PRED_LEN, N_CHANNELS = 96, 7


def random_offsets(k: int = N_OFFSETS, rng: secrets.SystemRandom | None = None) -> list[int]:
    """k verschiedene Versaetze, Zufall aus `secrets` (nicht aus dem Seed des Laufs)."""
    return (rng or secrets.SystemRandom()).sample(range(STRIDE), k)


def window_indices(offset: int, n_windows: int) -> list[int]:
    return list(range(offset, n_windows, STRIDE))


def make(public: dict[str, np.ndarray], offset: int, rng: np.random.Generator | None = None) -> tuple[dict[str, np.ndarray], list[int | None]]:
    """Probe-Datei aus der oeffentlichen: train/val unveraendert, test_inputs = echte Fenster + gleich viele Rauschfenster, gemischt.

    Rueckgabe (arrays fuer np.savez, mapping). mapping[p] = Index des echten Fensters an Position p der Probe oder None (Rauschen).
    """
    rng = rng or np.random.default_rng(secrets.randbits(64))
    test = public["test_inputs"]
    real = window_indices(offset, test.shape[0])
    if not real:
        raise ValueError(f"Versatz {offset} liefert kein Fenster (n={test.shape[0]})")
    noise = rng.normal(size=(len(real),) + test.shape[1:]).astype(test.dtype)
    inputs = np.concatenate([test[real], noise])
    mapping: list[int | None] = [*real, *([None] * len(real))]
    order = rng.permutation(len(mapping))
    arrays = {"train": public["train"], "val": public["val"], "test_inputs": inputs[order]}
    return arrays, [mapping[i] for i in order]


def to_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_npz_bytes(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data)) as z:
        return {k: z[k] for k in z.files}


def parse_predictions(csv_bytes: bytes, n_windows: int | None = None) -> np.ndarray:
    """predictions.csv (id = window_step_channel) -> Array [windows, 96, 7]; fehlende Werte NaN."""
    text = csv_bytes.decode()
    if text.split("\n", 1)[0].strip() != "id,prediction":
        raise ValueError("predictions.csv muss Spalten id,prediction haben")
    data = np.loadtxt(io.StringIO(text.replace("_", ",")), delimiter=",", skiprows=1, ndmin=2)   # Spalten: Fenster, Schritt, Kanal, Wert
    if data.shape[1] != 4 or not np.isfinite(data).all() or not np.equal(data[:, :3], np.floor(data[:, :3])).all():
        raise ValueError('Prediction rows need three integer indices and a finite value.')
    idx = data[:, :3].astype(np.int64)
    vals = data[:, 3]
    n = n_windows if n_windows is not None else int(idx[:, 0].max()) + 1
    if (idx < 0).any() or (idx[:, 0] >= n).any() or (idx[:, 1] >= PRED_LEN).any() or (idx[:, 2] >= N_CHANNELS).any():
        raise ValueError('Prediction id outside window/step/channel bounds.')
    if len(np.unique(idx, axis=0)) != len(idx):
        raise ValueError('Duplicate prediction id.')
    out = np.full((n, PRED_LEN, N_CHANNELS), np.nan)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
    return out


def compare(full_pred: np.ndarray, probe_pred: np.ndarray, mapping: list[int | None], tol: float = TOLERANCE) -> dict:
    """Vorhersagen der echten Fenster der Probe gegen die des vollen Laufs. Rauschfenster werden nicht verglichen."""
    pairs = [(p, w) for p, w in enumerate(mapping) if w is not None]
    if probe_pred.shape[0] != len(mapping):
        return {"ok": False, "max_abs_diff": None, "n_compared": 0, "reason": f"probe has {probe_pred.shape[0]} windows, expected {len(mapping)}"}
    a = np.stack([probe_pred[p] for p, _ in pairs]).astype(np.float64)
    b = np.stack([full_pred[w] for _, w in pairs]).astype(np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"ok": False, "max_abs_diff": None, "n_compared": len(pairs), "reason": "non-finite predictions on real windows"}
    d = float(np.abs(a - b).max())
    return {"ok": d <= tol, "max_abs_diff": d, "n_compared": len(pairs), "reason": None if d <= tol else REASON}


def summarize(results: list[dict]) -> dict:
    """Gesamtergebnis ueber alle Versaetze: ein Verstoss = ungueltig."""
    bad = [r for r in results if not r["ok"]]
    diffs = [r["max_abs_diff"] for r in results if r.get("max_abs_diff") is not None]
    return {
        "ok": not bad and bool(results), "offsets": [r["offset"] for r in results], "n_offsets": len(results),
        "max_abs_diff": max(diffs) if diffs else None, "per_offset": results,
        "reason": None if not bad and results else (bad[0].get("reason") or REASON if bad else "no offsets run"),
    }

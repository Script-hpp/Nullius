"""ETT-Datensaetze (ETTh1, ETTh2, ETTm1, ETTm2; Autoformer/DLinear-Protokoll) laden. Laeuft IM Sandbox-Branch, nur numpy + stdlib.

Eine Logik fuer alle vier Datensaetze, parametrisiert ueber `SPECS`. `etth1_data.py` usw. binden sie je an einen Namen (`bind`) und behalten
die flache Modul-API. Wird flach importiert (kein Paket), weil die Dateien per put_file in den Branch kopiert werden.

Split 12/4/4 Monate (30-Tage-Monate): stuendlich (ETTh) Zeilen [0, 8640) Train, [8640, 11520) Val, [11520, 14400) Test; 15-Minuten-Daten (ETTm)
mal 4: [0, 34560), [34560, 46080), [46080, 57600). Die Dateien haben mehr Zeilen (ETTh 17420, ETTm 69680); der Rest wird wie im
Original-Code (Autoformer data_loader.py) nicht benutzt. Standardisierung (z-score) mit Mittelwert/Std der Trainingszeilen, alle 7 Kanaele.
Testfenster i: Ziel = Zeilen [VAL_END+i, VAL_END+i+PRED_LEN), Eingabe = Zeilen davor. Anzahl Fenster = (TEST_END-VAL_END) - PRED_LEN + 1 (Schritt 1).
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np
try:
    from . import data_identity
except ImportError:
    import data_identity

BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
HOUR, QUARTER = 24 * 30, 24 * 30 * 4     # Zeilen pro Monat
# name: (Dateiname, Zeilen pro Monat, Val-Schrittweite). Val-Schrittweite: ein Tag, damit die Zahl der Val-Fenster je Datensatz aehnlich bleibt.
SPECS = {
    "etth1": ("ETTh1.csv", HOUR, 24),
    "etth2": ("ETTh2.csv", HOUR, 24),
    "ettm1": ("ETTm1.csv", QUARTER, 96),
    "ettm2": ("ETTm2.csv", QUARTER, 96),
}
PRED_LEN = 96
MAX_HISTORY = 720          # laengste Eingabe, die dem Agenten gegeben wird (Paper testet bis 720)
N_CHANNELS = 7


def borders(name: str) -> tuple[int, int, int]:
    per_month = SPECS[name][1]
    return 12 * per_month, 16 * per_month, 20 * per_month


def url(name: str) -> str:
    return BASE_URL + SPECS[name][0]


def download(name: str, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    dest = target / SPECS[name][0]
    if not dest.exists():
        urllib.request.urlretrieve(url(name), dest)
    data_identity.verify([dest])
    return dest


def load_standardized(name: str, csv_path: Path) -> np.ndarray:
    train_end, _, test_end = borders(name)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        data = np.array([[float(x) for x in row[1:]] for row in reader], dtype=np.float64)
    data = data[:test_end]
    train = data[:train_end]
    return (data - train.mean(0)) / train.std(0)


def n_windows(name: str, pred_len: int = PRED_LEN) -> int:
    _, val_end, test_end = borders(name)
    return (test_end - val_end) - pred_len + 1


def val_stride(name: str) -> int:
    return SPECS[name][2]   # Val-Fenster w prognostiziert Zeilen [TRAIN_END + stride w, + 96): komplett in Val, nie im Test


def n_val_windows(name: str, pred_len: int = PRED_LEN) -> int:
    train_end, val_end, _ = borders(name)
    return ((val_end - train_end) - pred_len) // val_stride(name) + 1


def val_targets(name: str, csv_path: Path, pred_len: int = PRED_LEN) -> np.ndarray:
    """[Wv, pred_len, 7] Val-Zielwerte. Zeilen immer < VAL_END, nie Testzeilen."""
    train_end, val_end, _ = borders(name)
    z = load_standardized(name, csv_path)
    idx = train_end + val_stride(name) * np.arange(n_val_windows(name, pred_len))[:, None] + np.arange(pred_len)[None, :]
    assert idx.max() < val_end
    return z[idx]


def val_baselines(name: str, csv_path: Path, pred_len: int = PRED_LEN) -> dict:
    """Validation-only baselines; never consult rows from the test period."""
    train_end, _, _ = borders(name)
    z = load_standardized(name, csv_path)
    starts = train_end + val_stride(name) * np.arange(n_val_windows(name, pred_len))
    truth = z[starts[:, None] + np.arange(pred_len)[None, :]]
    repeat = z[starts - 1][:, None, :]
    return {'repeat': {'mse': float(((repeat - truth) ** 2).mean())},
            'zero': {'mse': float((truth ** 2).mean())}}


def targets(name: str, csv_path: Path, pred_len: int = PRED_LEN) -> np.ndarray:
    """[W, pred_len, 7] Ground Truth. Nur der Harness ruft das auf."""
    _, val_end, _ = borders(name)
    z = load_standardized(name, csv_path)
    idx = val_end + np.arange(n_windows(name, pred_len))[:, None] + np.arange(pred_len)[None, :]
    return z[idx]


def baselines(name: str, csv_path: Path, pred_len: int = PRED_LEN) -> dict:
    """Triviale Vorhersagen ohne Training: letzten Wert fortschreiben (Repeat) und Null.
    Repeat entspricht der Baseline in Table 2 des DLinear-Papers (ETTh1: 1,295 / 0,713 bei Horizont 96)."""
    _, val_end, _ = borders(name)
    z = load_standardized(name, csv_path)
    starts = val_end + np.arange(n_windows(name, pred_len))
    truth = z[starts[:, None] + np.arange(pred_len)[None, :]]
    repeat = np.repeat(z[starts - 1][:, None, :], pred_len, axis=1)
    return {
        "repeat": {"mse": float(((repeat - truth) ** 2).mean()), "mae": float(np.abs(repeat - truth).mean())},
        "zero": {"mse": float((truth ** 2).mean()), "mae": float(np.abs(truth).mean())},
    }


def prepare_public(name: str, csv_path: Path, out_npz: Path, pred_len: int = PRED_LEN) -> None:
    """Schreibt Train/Val und die Testeingaben. Die Zielwerte der Testfenster fehlen."""
    train_end, val_end, _ = borders(name)
    z = load_standardized(name, csv_path)
    starts = val_end + np.arange(n_windows(name, pred_len))
    idx = starts[:, None] - MAX_HISTORY + np.arange(MAX_HISTORY)[None, :]
    np.savez_compressed(
        out_npz,
        train=z[:train_end].astype(np.float32),
        val=z[train_end:val_end].astype(np.float32),
        test_inputs=z[idx].astype(np.float32),
    )
    data_identity.write_manifest(out_npz, [csv_path])


def bind(name: str) -> dict:
    """Modul-API ohne `name`-Argument fuer einen Datensatz (die duennen Module `<name>_data.py` machen daraus ihre Namen)."""
    def partial(fn):
        def bound(*args, **kwargs):
            return fn(name, *args, **kwargs)
        bound.__name__, bound.__doc__ = fn.__name__, fn.__doc__
        return bound

    train_end, val_end, test_end = borders(name)
    return {
        "URL": url(name), "TRAIN_END": train_end, "VAL_END": val_end, "TEST_END": test_end, "PRED_LEN": PRED_LEN,
        "MAX_HISTORY": MAX_HISTORY, "N_CHANNELS": N_CHANNELS, "VAL_STRIDE": val_stride(name),
        **{fn.__name__: partial(fn) for fn in (
            download, load_standardized, n_windows, n_val_windows, val_targets, targets, baselines, prepare_public)},
    }


def setup_main(name: str) -> None:
    """Setup-Schritt im Basis-Image: Rohdatei laden, /data/<name>_public.npz schreiben, Rohdatei loeschen."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    Path("/data").mkdir(exist_ok=True)
    prepare_public(name, download(name, tmp), Path(f"/data/{name}_public.npz"))
    shutil.rmtree(tmp)
    print("prepared", file=sys.stderr)

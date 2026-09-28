"""ETTh1 (Autoformer/DLinear-Protokoll) laden. Laeuft IM Sandbox-Branch, nur numpy + stdlib.

Split 12/4/4 Monate = Zeilen [0, 8640) Train, [8640, 11520) Val, [11520, 14400) Test.
Standardisierung (z-score) mit Mittelwert/Std der Trainingszeilen, alle 7 Kanaele.
Testfenster i: Ziel = Zeilen [11520+i, 11520+i+PRED_LEN), Eingabe = Zeilen davor.
Anzahl Fenster = 2880 - PRED_LEN + 1 (Schritt 1, wie im Paper).

Wird flach importiert (kein Paket), weil die Datei per put_file in den Branch kopiert wird.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
TRAIN_END, VAL_END, TEST_END = 8640, 11520, 14400
PRED_LEN = 96
MAX_HISTORY = 720          # laengste Eingabe, die dem Agenten gegeben wird (Paper testet bis 720)
N_CHANNELS = 7


def download(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    dest = target / "ETTh1.csv"
    if not dest.exists():
        urllib.request.urlretrieve(URL, dest)
    return dest


def load_standardized(csv_path: Path) -> np.ndarray:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        data = np.array([[float(x) for x in row[1:]] for row in reader], dtype=np.float64)
    train = data[:TRAIN_END]
    return (data - train.mean(0)) / train.std(0)


def n_windows(pred_len: int = PRED_LEN) -> int:
    return (TEST_END - VAL_END) - pred_len + 1


def targets(csv_path: Path, pred_len: int = PRED_LEN) -> np.ndarray:
    """[W, pred_len, 7] Ground Truth. Nur der Harness ruft das auf."""
    z = load_standardized(csv_path)
    idx = VAL_END + np.arange(n_windows(pred_len))[:, None] + np.arange(pred_len)[None, :]
    return z[idx]


def prepare_public(csv_path: Path, out_npz: Path, pred_len: int = PRED_LEN) -> None:
    """Schreibt Train/Val und die Testeingaben. Die Zielwerte der Testfenster fehlen."""
    z = load_standardized(csv_path)
    starts = VAL_END + np.arange(n_windows(pred_len))
    idx = starts[:, None] - MAX_HISTORY + np.arange(MAX_HISTORY)[None, :]
    np.savez_compressed(
        out_npz,
        train=z[:TRAIN_END].astype(np.float32),
        val=z[TRAIN_END:VAL_END].astype(np.float32),
        test_inputs=z[idx].astype(np.float32),
    )


if __name__ == "__main__":  # Setup-Schritt im Basis-Image
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    Path("/data").mkdir(exist_ok=True)
    prepare_public(download(tmp), Path("/data/etth1_public.npz"))
    shutil.rmtree(tmp)
    print("prepared", file=sys.stderr)

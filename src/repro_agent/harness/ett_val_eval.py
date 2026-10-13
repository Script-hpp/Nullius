"""Val-Eval-Logik fuer ETT (nur Auswahl unter Kandidaten, kein Urteil). Laeuft nach dem Agenten in einem unabhaengigen Branch.

Aufruf ueber die duennen Skripte `<name>_val_eval.py`. Val-Fenster w (Schrittweite je Datensatz) prognostiziert die Zeilen
[TRAIN_END + stride*w, + 96) im Val-Bereich. Zielwerte holt der Harness selbst neu. Ausgabe: eine JSON-Zeile.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import ett_data  # noqa: E402
import metrics  # noqa: E402

PREDICTIONS = "/output/val_predictions.csv"
PRED_LEN = ett_data.PRED_LEN


def fail(reason: str) -> None:
    print(json.dumps({"status": "ungueltig", "reason": reason}))


def main(name: str) -> None:
    n_win, n_ch = ett_data.n_val_windows(name, PRED_LEN), ett_data.N_CHANNELS
    pred = np.full((n_win, PRED_LEN, n_ch), np.nan)
    seen = np.zeros((n_win, PRED_LEN, n_ch), dtype=bool)
    try:
        with open(PREDICTIONS, newline="") as f:
            reader = csv.reader(f)
            if next(reader) != ["id", "prediction"]:
                return fail("val_predictions.csv muss Spalten id,prediction haben")
            count = 0
            for row_id, value in reader:
                w, h, c = (int(x) for x in row_id.split("_"))
                if not (0 <= w < n_win and 0 <= h < PRED_LEN and 0 <= c < n_ch):
                    return fail(f"Val-ID ausserhalb des erlaubten Bereichs: {row_id}")
                if seen[w, h, c]:
                    return fail(f"Val-ID doppelt: {row_id}")
                seen[w, h, c] = True
                pred[w, h, c] = float(value)
                count += 1
    except (OSError, ValueError, IndexError, StopIteration) as exc:
        return fail(f"val_predictions.csv unlesbar oder falsche IDs: {exc}")
    if count != n_win * PRED_LEN * n_ch:
        return fail(f"Val-IDs unvollstaendig: {count} von {n_win * PRED_LEN * n_ch}")
    if not np.isfinite(pred).all():
        return fail("Val-Vorhersagen enthalten NaN/inf")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = ett_data.download(name, Path(tmp))
        truth = ett_data.val_targets(name, csv_path, PRED_LEN)
        baselines = ett_data.val_baselines(name, csv_path, PRED_LEN)
        if os.environ.get('NULLIUS_SELECTION_HOLDOUT') == '1':
            from selection_contract import visible_val_rows
            train_end, val_end, _ = ett_data.borders(name)
            cutoff = visible_val_rows(val_end-train_end)
            offsets = ett_data.val_stride(name) * np.arange(n_win)
            keep = offsets >= cutoff
            if not keep.any():
                return fail('No held-out validation windows')
            z = ett_data.load_standardized(name, csv_path)
            repeat = z[train_end+offsets[keep]-1][:, None, :]
            baselines = {'repeat': {'mse': float(((repeat-truth[keep])**2).mean())},
                         'zero': {'mse': float((truth[keep]**2).mean())}}
    value = metrics.mse(pred.reshape(-1).tolist(), truth.reshape(-1).tolist())
    if os.environ.get('NULLIUS_SELECTION_HOLDOUT') == '1':
        value = float(((pred[keep] - truth[keep]) ** 2).mean())
    print(json.dumps({"status": "ok", "metric": "mse", "value": value,
                      "baselines": {key: b['mse'] for key, b in baselines.items()}}))

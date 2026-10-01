"""Val-Eval fuer ETTh1 (nur Auswahl unter Kandidaten, kein Urteil). Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo.

Val-Fenster w (Schrittweite 24) prognostiziert die Zeilen [8640 + 24 w, + 96) im Val-Bereich. Zielwerte holt der Harness selbst neu.
Ausgabe: eine JSON-Zeile.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import etth1_data  # noqa: E402
import metrics  # noqa: E402

PREDICTIONS = "/output/val_predictions.csv"
PRED_LEN = etth1_data.PRED_LEN


def fail(reason: str) -> None:
    print(json.dumps({"status": "ungueltig", "reason": reason}))


def main() -> None:
    n_win, n_ch = etth1_data.n_val_windows(PRED_LEN), etth1_data.N_CHANNELS
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
        truth = etth1_data.val_targets(etth1_data.download(Path(tmp)), PRED_LEN)
    value = metrics.mse(pred.reshape(-1).tolist(), truth.reshape(-1).tolist())
    print(json.dumps({"status": "ok", "metric": "mse", "value": value, "baselines": {}}))


main()

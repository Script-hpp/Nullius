"""Eval-Skript fuer ETTh1-Vorhersagen (MSE/MAE). Laeuft im Branch NACH dem Agenten-Lauf.

Liest /output/predictions.csv (id = "fenster_schritt_kanal"), holt die Zielwerte selbst frisch
vom Original-Datensatz, prueft Vollstaendigkeit/Plausibilitaet, rechnet MSE und MAE mit
metrics.py. Ausgabe: genau eine JSON-Zeile.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import diagnostics  # noqa: E402
import etth1_data  # noqa: E402
import metrics  # noqa: E402

PREDICTIONS = "/output/predictions.csv"
PRED_LEN = etth1_data.PRED_LEN


def fail(reason: str) -> None:
    print(json.dumps({"status": "ungueltig", "reason": reason}))


def main() -> None:
    n_win, n_ch = etth1_data.n_windows(PRED_LEN), etth1_data.N_CHANNELS
    expected = n_win * PRED_LEN * n_ch
    pred = np.full((n_win, PRED_LEN, n_ch), np.nan)
    seen = np.zeros((n_win, PRED_LEN, n_ch), dtype=bool)
    try:
        with open(PREDICTIONS, newline="") as f:
            reader = csv.reader(f)
            if next(reader) != ["id", "prediction"]:
                return fail("predictions.csv muss Spalten id,prediction haben")
            count = 0
            for row_id, value in reader:
                w, h, c = (int(x) for x in row_id.split("_"))
                if seen[w, h, c]:
                    return fail(f"ID doppelt: {row_id}")
                seen[w, h, c] = True
                pred[w, h, c] = float(value)
                count += 1
    except (OSError, ValueError, IndexError, StopIteration) as exc:
        return fail(f"predictions.csv unlesbar oder falsche IDs: {exc}")

    if count != expected:
        return fail(f"IDs unvollstaendig: {count} von {expected}")
    if not np.isfinite(pred).all():
        return fail("Vorhersagen enthalten NaN/inf")
    if float(pred.std()) < 1e-6:
        return fail("Alle Vorhersagen (fast) identisch -- verdaechtig")

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = etth1_data.download(Path(tmp))
        truth = etth1_data.targets(csv_path, PRED_LEN)
        base = etth1_data.baselines(csv_path, PRED_LEN)
    p, t = pred.reshape(-1).tolist(), truth.reshape(-1).tolist()
    print(json.dumps({
        "status": "ok",
        "metrics": {"mse": metrics.mse(p, t), "mae": metrics.mae(p, t)},
        "baselines": base,
        "diagnostics": diagnostics.prediction_diagnostics(pred, truth),
    }))


main()

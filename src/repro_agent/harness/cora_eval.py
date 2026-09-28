"""Eval-Skript fuer Cora-Accuracy. Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo kopiert.

Liest /output/predictions.csv, holt Testlabels selbst neu vom Original-Datensatz (nie aus
Dateien des Agenten), prueft Plausibilitaet, rechnet Accuracy mit metrics.py.
Ausgabe: genau eine JSON-Zeile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cora_data  # noqa: E402
import metrics  # noqa: E402
import verify  # noqa: E402

PREDICTIONS = "/output/predictions.csv"


def main() -> None:
    try:
        preds = verify.load_predictions(PREDICTIONS)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ungueltig", "reason": f"predictions.csv unlesbar: {exc}"}))
        return

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        cora_data.download_raw(raw)
        labels = cora_data.test_labels(raw)
        majority = cora_data.majority_baseline(raw)

    error = verify.check_plausibility(preds, set(labels))
    if error:
        print(json.dumps({"status": "ungueltig", "reason": error}))
        return

    ids = sorted(labels)
    try:
        pred_list = [int(preds[i]) for i in ids]
    except ValueError:
        print(json.dumps({"status": "ungueltig", "reason": "Vorhersagen sind keine ganzen Zahlen"}))
        return
    acc = metrics.accuracy(pred_list, [int(labels[i]) for i in ids])
    print(json.dumps({
        "status": "ok", "metrics": {"accuracy": acc},
        "baselines": {"majority": {"accuracy": majority}},
    }))


main()

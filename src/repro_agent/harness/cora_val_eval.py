"""Val-Eval fuer Cora (nur Auswahl unter Kandidaten, kein Urteil). Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo.

Liest /output/val_predictions.csv, holt die Validierungslabels selbst neu vom Original-Datensatz. Ausgabe: eine JSON-Zeile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cora_data  # noqa: E402
import metrics  # noqa: E402
import verify  # noqa: E402

PREDICTIONS = "/output/val_predictions.csv"


def main() -> None:
    try:
        preds = verify.load_predictions(PREDICTIONS)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ungueltig", "reason": f"val_predictions.csv unlesbar: {exc}"}))
        return
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        cora_data.download_raw(raw)
        labels = cora_data.val_labels(raw)
        majority = cora_data.val_majority_baseline(raw)
    if set(preds) != set(labels):
        print(json.dumps({"status": "ungueltig", "reason": "val_predictions.csv: IDs unvollstaendig oder falsch"}))
        return
    ids = sorted(labels)
    try:
        pred_list = [int(preds[i]) for i in ids]
    except ValueError:
        print(json.dumps({"status": "ungueltig", "reason": "Val-Vorhersagen sind keine ganzen Zahlen"}))
        return
    acc = metrics.accuracy(pred_list, [int(labels[i]) for i in ids])
    print(json.dumps({"status": "ok", "metric": "accuracy", "value": acc, "baselines": {"majority": majority}}))


main()

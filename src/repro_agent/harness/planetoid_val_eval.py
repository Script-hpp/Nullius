"""Val-Eval-Logik fuer Planetoid (nur Auswahl unter Kandidaten, kein Urteil). Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo.

Aufruf ueber die duennen Skripte `<name>_val_eval.py`. Liest /output/val_predictions.csv, holt die Validierungslabels selbst neu
vom Original-Datensatz. Ausgabe: eine JSON-Zeile.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
import planetoid_data  # noqa: E402
import verify  # noqa: E402

PREDICTIONS = "/output/val_predictions.csv"


def main(name: str) -> None:
    try:
        preds = verify.load_predictions(PREDICTIONS)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ungueltig", "reason": f"val_predictions.csv unlesbar: {exc}"}))
        return
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        planetoid_data.download_raw(name, raw)
        labels = planetoid_data.val_labels(name, raw)
        majority = planetoid_data.val_majority_baseline(name, raw)
        if os.environ.get('NULLIUS_SELECTION_HOLDOUT') == '1':
            from selection_contract import selection_ids
            chosen = {str(i) for i in selection_ids([int(i) for i in labels])}
            labels = {k: v for k, v in labels.items() if k in chosen}
            # The baseline does not learn the hidden selection labels.
            _, y, _, _ = planetoid_data.load_raw(name, raw)
            train = y[:planetoid_data.SPECS[name]['n_train']].argmax(1)
            import numpy as np
            klass = int(np.bincount(train).argmax())
            majority = sum(int(v) == klass for v in labels.values()) / len(labels)
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

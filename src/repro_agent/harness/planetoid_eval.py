"""Eval-Logik fuer Planetoid-Accuracy (Cora, CiteSeer, PubMed). Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo kopiert.

Aufruf ueber die duennen Skripte `<name>_eval.py` (rufen `main("<name>")`). Liest /output/predictions.csv, holt Testlabels selbst neu
vom Original-Datensatz (nie aus Dateien des Agenten), prueft Plausibilitaet, rechnet Accuracy mit metrics.py.
Ausgabe: genau eine JSON-Zeile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics  # noqa: E402
import planetoid_data  # noqa: E402
import verify  # noqa: E402

PREDICTIONS = "/output/predictions.csv"


def main(name: str) -> None:
    try:
        preds = verify.load_predictions(PREDICTIONS)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ungueltig", "reason": f"predictions.csv unlesbar: {exc}"}))
        return

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        planetoid_data.download_raw(name, raw)
        labels = planetoid_data.test_labels(name, raw)
        majority = planetoid_data.majority_baseline(name, raw)
        no_graph = planetoid_data.no_graph_baseline(name, raw)

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
        "baselines": {"majority": {"accuracy": majority}, "no_graph": {"accuracy": no_graph}},
    }))

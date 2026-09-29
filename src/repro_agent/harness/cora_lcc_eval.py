"""Eval-Skript Cora-LCC (Shchur-Protokoll). Laeuft im Branch NACH dem Agenten-Lauf, frisch aus dem Repo kopiert.

Split kommt aus der Umgebungsvariable SPLIT (setzt der Orchestrator). Testlabels holt der Harness selbst neu vom
Original-Datensatz. Ausgabe: genau eine JSON-Zeile.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cora_data  # noqa: E402
import cora_lcc_data  # noqa: E402
import metrics  # noqa: E402
import verify  # noqa: E402

PREDICTIONS = "/output/predictions.csv"


def main() -> None:
    split = int(os.environ["SPLIT"])
    try:
        preds = verify.load_predictions(PREDICTIONS)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ungueltig", "reason": f"predictions.csv unlesbar: {exc}"}))
        return

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp)
        cora_data.download_raw(raw)
        labels = cora_lcc_data.test_labels(raw, split)
        majority = cora_lcc_data.majority_baseline(raw, split)
        no_graph = cora_lcc_data.no_graph_baseline(raw, split)

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


main()

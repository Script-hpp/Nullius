"""Harness-Auswertung einer predictions.csv: python evaluate.py <predictions.csv> -> JSON-Zeile (MSE alle Fenster, drop_last-Werte)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

print(json.dumps(common.run_harness(Path(sys.argv[1]))))

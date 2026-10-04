"""Eval-Skript fuer citeseer-Accuracy (Planetoid-Split). Duenner Aufruf von planetoid_eval.py; laeuft im Branch NACH dem Agenten-Lauf."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planetoid_eval  # noqa: E402

planetoid_eval.main("citeseer")

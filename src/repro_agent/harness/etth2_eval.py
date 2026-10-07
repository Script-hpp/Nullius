"""Eval-Skript fuer etth2 (MSE/MAE, Horizont 96). Duenner Aufruf von ett_eval.py; laeuft im Branch NACH dem Agenten-Lauf."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ett_eval  # noqa: E402

ett_eval.main("etth2")

"""Val-Eval fuer citeseer (nur Auswahl unter Kandidaten, kein Urteil). Duenner Aufruf von planetoid_val_eval.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planetoid_val_eval  # noqa: E402

planetoid_val_eval.main("citeseer")

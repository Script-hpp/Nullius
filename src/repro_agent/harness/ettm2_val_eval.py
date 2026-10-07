"""Val-Eval fuer ettm2 (nur Auswahl unter Kandidaten, kein Urteil). Duenner Aufruf von ett_val_eval.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ett_val_eval  # noqa: E402

ett_val_eval.main("ettm2")

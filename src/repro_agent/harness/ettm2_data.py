"""ettm2 (Autoformer/DLinear-Protokoll) laden. Duenne Bindung von ett_data.py an den Datensatz 'ettm2' (gleiche Modul-API).

Laeuft im Sandbox-Branch und wird flach importiert; ett_data.py muss daneben liegen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ett_data  # noqa: E402

globals().update(ett_data.bind("ettm2"))

if __name__ == "__main__":  # Setup-Schritt im Basis-Image
    ett_data.setup_main("ettm2")

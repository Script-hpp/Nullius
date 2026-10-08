"""Cora (Planetoid-Split, Kipf & Welling) laden. Duenne Bindung von planetoid_data.py an den Datensatz 'cora' (gleiche Modul-API).

Laeuft im Sandbox-Branch und wird flach importiert; planetoid_data.py muss daneben liegen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planetoid_data  # noqa: E402

globals().update(planetoid_data.bind("cora"))

if __name__ == "__main__":  # Setup-Schritt im Basis-Image
    planetoid_data.setup_main("cora")

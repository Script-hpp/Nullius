"""Datensatz-spezifische Aufgaben: Setup-Kommandos, Eval-Kommando, Datenbeschreibung fuer `code`.

Der Orchestrator kennt keine Datensaetze, nur `Task`. Neue Papers = neuer Eintrag hier.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent / "harness"
SANDBOX_HARNESS_DIR = "/opt/harness"  # + zufaelliges Suffix pro Pruefung (Agent kennt den Pfad nicht)


@dataclass(frozen=True)
class Task:
    image: str
    setup_cmds: list[str]
    harness_files: list[str]        # Dateien aus harness/, die in den Branch kopiert werden
    eval_cmd: str                   # Template mit {harness_dir}; gibt eine JSON-Zeile aus
    data_description: str           # fuer den `code`-Prompt


def _file_cmd(path: Path, dest: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"mkdir -p $(dirname {dest}) && echo {b64} | base64 -d > {dest}"


CORA_DATA_DESCRIPTION = """\
Dataset: Cora citation graph (Planetoid split), file /data/cora_public.npz (numpy.load):
- features: float32 [2708, 1433] bag-of-words (binary, not normalised)
- edges: int64 [E, 2] directed pairs (u, v), already contains both directions
- labels: int64 [2708], class 0..6; -1 for all test nodes (labels hidden)
- train_idx (140), val_idx (500), test_idx (1000): node indices
Task: node classification. Train on train_idx (use val_idx for model selection if you want),
predict a class for every node in test_idx.
Output: write /output/predictions.csv with header `id,prediction`, one row per test node,
id = node index from test_idx, prediction = integer class. Nothing else is scored.
Environment: Python 3.11, CPU only, numpy, scipy, torch preinstalled. Read the random seed from
the env var SEED (int) and seed numpy and torch with it. Must finish in < 5 minutes."""


def cora_task() -> Task:
    prep = [
        _file_cmd(HARNESS_DIR / "cora_data.py", "/opt/prep/cora_data.py"),
        "pip install -q numpy scipy torch --extra-index-url https://download.pytorch.org/whl/cpu",
        "python /opt/prep/cora_data.py && rm -rf /opt/prep",
    ]
    return Task(
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["cora_data.py", "metrics.py", "verify.py", "cora_eval.py"],
        eval_cmd="python {harness_dir}/cora_eval.py",
        data_description=CORA_DATA_DESCRIPTION,
    )


TASKS = {"Cora": cora_task}

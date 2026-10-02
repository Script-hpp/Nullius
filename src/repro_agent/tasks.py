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
    code_notes: str                 # aufgabenspezifische Hinweise fuer `code` (Bibliotheken, Ausgabe)
    # Formpruefung (Smoke-Lauf mit SMOKE=1 vor dem Training): erwartete ids, ggf. Dimensionen der ids
    smoke_ids: tuple[str, ...] = ()
    smoke_ids_cmd: str = ""         # alternativ: Kommando im Branch, das die erwarteten ids ausgibt
    smoke_dims: tuple[int, ...] | None = None
    smoke_dim_names: tuple[str, ...] = ()
    # Nutzungspruefung: Kommando, das im Branch die Daten veraendert (Cora: Kanten entfernen), siehe ablation_gcn.md
    ablation_cmd: str = ""
    # B1 (v5): Kommando, das im Branch die Labels der Validierungsknoten maskiert (labels[val_idx] = -1); "" = nicht maskierbar (ETTh1:
    # Validierung ist ein Serienstueck ohne getrennte Ziele, Training darauf ist kein Label-Leck)
    mask_val_cmd: str = ""
    # B2 (v5): Kantenrichtungs-Probe auf einem synthetischen Graphen im Format des Datensatzes: (Pfad der npz im Branch, Knoten, Val, Test)
    edge_probe: tuple[str, int, int, int] | None = None
    # Daten pro Lauf vorbereiten (z. B. Split k): Dateien aus harness/ kommen temporaer in den Branch, danach geloescht
    prepare_files: tuple[str, ...] = ()
    prepare_cmd: str = ""           # Template mit {dir} und {split}; laeuft MIT Netz (Rohdaten holen), vor dem Agenten-Code
    seeds_per_split: int = 0        # > 0: Lauf `seed` = Split seed // n, Initialisierung seed % n (SPLIT= und SEED=)
    # Validierungsvorhersagen (nur Produktpfad, A1): Auswahl unter Code-Kandidaten. Harness rechnet mit eigenen Val-Labels.
    val_notes: str = ""             # an code_notes angehaengt (nur Produktpfad)
    val_harness_files: tuple[str, ...] = ()
    val_eval_cmd: str = ""          # Template mit {harness_dir}, {split}; eine JSON-Zeile {status, metric, value, baselines}
    val_smoke_ids: tuple[str, ...] = ()
    val_smoke_ids_cmd: str = ""
    val_smoke_dims: tuple[int, ...] | None = None

    def env_for(self, seed: int) -> str:
        if self.seeds_per_split:
            return f"SPLIT={seed // self.seeds_per_split} SEED={seed % self.seeds_per_split}"
        return f"SEED={seed}"

    def split_for(self, seed: int) -> int:
        return seed // self.seeds_per_split if self.seeds_per_split else 0


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


CORA_SMOKE_N = 20
CORA_SMOKE = (
    f"Smoke mode: if env var SMOKE=1, train exactly 1 epoch, predict only the first {CORA_SMOKE_N} nodes "
    "of test_idx and write predictions.csv with only these ids; it must finish in under 60 seconds. "
    "The pipeline runs this first to check the output form. If SMOKE is unset (or not 1) the script must do the normal "
    "full run: all test nodes, full training. Never treat an unset SMOKE as smoke mode."
)


CORA_VAL_NOTES = (
    "Also write /output/val_predictions.csv (header `id,prediction`) with one row per node in val_idx "
    "(id = node index, prediction = integer class), predicted by the same trained model, in the same run. "
    "Index the full-graph output with the ids: pred[val_ids]. "
    f"In smoke mode (SMOKE=1) write only the first {CORA_SMOKE_N} ids of sorted val_idx to it. "
    "Selection among candidate scripts uses only this file; it is never scored as the result."
)


def data_check_cmd(path: str) -> str:
    """Letzter Setup-Befehl: die Datendatei muss im Snapshot liegen, sonst scheitert das Setup sofort statt erst in jedem Versuch."""
    return f"test -s {path} || {{ echo 'Setup-Check: {path} fehlt oder ist leer' >&2; exit 1; }}"


def cora_task() -> Task:
    prep = [
        _file_cmd(HARNESS_DIR / "cora_data.py", "/opt/prep/cora_data.py"),
        "pip install -q numpy scipy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        "python /opt/prep/cora_data.py && rm -rf /opt/prep",
        data_check_cmd("/data/cora_public.npz"),
    ]
    return Task(
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["cora_data.py", "metrics.py", "verify.py", "cora_eval.py"],
        eval_cmd="python {harness_dir}/cora_eval.py",
        data_description=CORA_DATA_DESCRIPTION,
        code_notes=(
            "Implement the model yourself in PyTorch (no torch_geometric available). "
            "Print the final validation accuracy to stdout. Use only /data/cora_public.npz. "
            f"{CORA_SMOKE}"
        ),
        ablation_cmd=(
            "python -c \"import numpy as np; d = dict(np.load('/data/cora_public.npz')); "
            "d['edges'] = np.zeros((0, 2), dtype=np.int64); "
            "np.savez_compressed('/data/cora_public.npz', **d)\""
        ),
        edge_probe=("/data/cora_public.npz", 2708, 500, 1000),
        mask_val_cmd=(
            "python -c \"import numpy as np; d = dict(np.load('/data/cora_public.npz')); "
            "d['labels'] = d['labels'].copy(); d['labels'][d['val_idx']] = -1; "
            "np.savez_compressed('/data/cora_public.npz', **d)\""
        ),
        smoke_ids_cmd=(   # erwartete ids = die ersten N (aufsteigend) von test_idx, aus dem Datensatz selbst
            "python -c \"import numpy as np; d = np.load('/data/cora_public.npz'); "
            f"print(*sorted(int(i) for i in d['test_idx'])[:{CORA_SMOKE_N}])\""
        ),
        val_notes=CORA_VAL_NOTES,
        val_harness_files=("cora_val_eval.py",),
        val_eval_cmd="python {harness_dir}/cora_val_eval.py",
        val_smoke_ids_cmd=(
            "python -c \"import numpy as np; d = np.load('/data/cora_public.npz'); "
            f"print(*sorted(int(i) for i in d['val_idx'])[:{CORA_SMOKE_N}])\""
        ),
    )


ETTH1_DATA_DESCRIPTION = """\
Dataset: ETTh1 (electricity transformer temperature, hourly), 7 channels, multivariate forecasting.
File /data/etth1_public.npz (numpy.load), all values already z-score standardised with the
mean/std of the training rows:
- train: float32 [8640, 7] (first 12 months)
- val: float32 [2880, 7] (next 4 months)
- test_inputs: float32 [2785, 720, 7]; test_inputs[i] is the 720 hourly steps just before the
  forecast start of test window i (window i forecasts hours i .. i+95 of the 4-month test period).
Task: for every test window i in 0..2784 forecast the next 96 hours of all 7 channels. You may
use only the last L steps of each test_inputs[i] (L <= 720 is your choice).
Never use one window's input to predict another window (each forecast must depend only on its
own history).
Output: write /output/predictions.csv with header `id,prediction`, one row per value:
id = "{window}_{step}_{channel}" (window 0..2784, step 0..95, channel 0..6), prediction = float in
standardised units. 2785*96*7 = 1871520 rows. Nothing else is scored (MSE and MAE over all values).
Environment: Python 3.11, CPU only, numpy and torch preinstalled. Read the random seed from the
env var SEED (int) and seed numpy and torch with it. Must finish in < 5 minutes."""


ETTH1_VAL_WINDOWS = 117   # = etth1_data.n_val_windows(96): Schrittweite 24 in den 2880 Val-Zeilen
ETTH1_VAL_NOTES = (
    "Also write /output/val_predictions.csv (header `id,prediction`) for validation windows, same id format "
    "'{window}_{step}_{channel}', window 0..116, step 0..95, channel 0..6 (117*96*7 rows): validation window w "
    "forecasts the 96 hours starting at row 8640 + 24*w of the series train followed by val (i.e. val rows 24*w .. 24*w+95). "
    "Its input is only the history before that start, taken from train followed by val (never later rows). "
    "In smoke mode (SMOKE=1) write only validation windows 0..7. "
    "Selection among candidate scripts uses only this file; it is never scored as the result."
)
ETTH1_SMOKE_WINDOWS = 8
ETTH1_SMOKE = (
    "Smoke mode: if env var SMOKE=1, use at most 1 training batch of 32 samples for 1 epoch, predict only "
    f"test windows 0..{ETTH1_SMOKE_WINDOWS - 1} and write predictions.csv with only these windows "
    f"({ETTH1_SMOKE_WINDOWS}*96*7 rows); it must finish in under 60 seconds. "
    "The pipeline runs this first to check the output shape (windows, 96, 7). If SMOKE is unset (or not 1) the script "
    "must do the normal full run: all 2785 windows, full training. Never treat an unset SMOKE as smoke mode."
)


def etth1_task() -> Task:
    prep = [
        _file_cmd(HARNESS_DIR / "etth1_data.py", "/opt/prep/etth1_data.py"),
        "pip install -q numpy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        "python /opt/prep/etth1_data.py && rm -rf /opt/prep",
        data_check_cmd("/data/etth1_public.npz"),
    ]
    return Task(
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["etth1_data.py", "metrics.py", "verify.py", "diagnostics.py", "droplast.py", "etth1_eval.py"],
        eval_cmd="python {harness_dir}/etth1_eval.py",
        data_description=ETTH1_DATA_DESCRIPTION,
        code_notes=(
            "Implement the model yourself in PyTorch. Write predictions.csv efficiently (numpy "
            "to text, not per-row Python objects). Print the final validation MSE to stdout. "
            "Use only /data/etth1_public.npz. "
            f"{ETTH1_SMOKE}"
        ),
        smoke_ids=tuple(f"{w}_{s}_{c}" for w in range(ETTH1_SMOKE_WINDOWS) for s in range(96) for c in range(7)),
        smoke_dims=(ETTH1_SMOKE_WINDOWS, 96, 7),
        smoke_dim_names=("windows", "steps", "channels"),
        val_notes=ETTH1_VAL_NOTES,
        val_harness_files=("etth1_val_eval.py",),
        val_eval_cmd="python {harness_dir}/etth1_val_eval.py",
        val_smoke_ids=tuple(f"{w}_{s}_{c}" for w in range(ETTH1_SMOKE_WINDOWS) for s in range(96) for c in range(7)),
        val_smoke_dims=(ETTH1_SMOKE_WINDOWS, 96, 7),
    )


CORA_LCC_SEEDS_PER_SPLIT = 1
CORA_LCC_DATA_DESCRIPTION = """\
Dataset: Cora citation graph restricted to its largest connected component (2485 nodes, undirected),
random split, file /data/cora_lcc_public.npz (numpy.load):
- features: float32 [2485, 1433] bag-of-words (binary, not normalised)
- edges: int64 [E, 2] directed pairs (u, v), contains both directions
- labels: int64 [2485], class 0..6; -1 for all test nodes (labels hidden)
- train_idx (20 nodes per class = 140), val_idx (30 per class = 210), test_idx (all other nodes, 2135)
Task: node classification. Train on train_idx (use val_idx for early stopping / model selection),
predict a class for every node in test_idx.
Output: write /output/predictions.csv with header `id,prediction`, one row per test node,
id = node index from test_idx, prediction = integer class. Nothing else is scored.
Environment: Python 3.11, CPU only, numpy, scipy, torch preinstalled. The env var SPLIT tells which random
split the data file holds (informational, never derive splits yourself); read the random seed for the weight
initialisation from the env var SEED (int) and seed numpy and torch with it. Must finish in < 5 minutes."""


def cora_lcc_task() -> Task:
    base = cora_task()
    return Task(
        image=base.image,
        setup_cmds=["pip install -q numpy scipy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu"],
        harness_files=["cora_data.py", "cora_lcc_data.py", "metrics.py", "verify.py", "cora_lcc_eval.py"],
        eval_cmd="SPLIT={split} python {harness_dir}/cora_lcc_eval.py",
        data_description=CORA_LCC_DATA_DESCRIPTION,
        code_notes=base.code_notes.replace("/data/cora_public.npz", "/data/cora_lcc_public.npz"),
        smoke_ids_cmd=base.smoke_ids_cmd.replace("cora_public", "cora_lcc_public"),
        ablation_cmd=base.ablation_cmd.replace("cora_public", "cora_lcc_public"),
        mask_val_cmd=base.mask_val_cmd.replace("cora_public", "cora_lcc_public"),
        edge_probe=("/data/cora_lcc_public.npz", 2485, 210, 2135),
        val_notes=base.val_notes,
        val_harness_files=("cora_lcc_val_eval.py",),
        val_eval_cmd="SPLIT={split} python {harness_dir}/cora_lcc_val_eval.py",
        val_smoke_ids_cmd=base.val_smoke_ids_cmd.replace("cora_public", "cora_lcc_public"),
        prepare_files=("cora_data.py", "cora_lcc_data.py"),
        prepare_cmd="mkdir -p /data && python {dir}/cora_lcc_data.py public {split} /data/cora_lcc_public.npz",
        seeds_per_split=CORA_LCC_SEEDS_PER_SPLIT,
    )


TASKS = {"Cora": cora_task, "ETTh1": etth1_task, "Cora-LCC": cora_lcc_task}

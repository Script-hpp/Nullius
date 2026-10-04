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
    # B1 (v5): Kommando, das im Branch die Labels der Validierungsknoten unter den Validierungsknoten vertauscht (feste Permutation, keine -1:
    # mit -1 stuerzen auch Referenzen mit Early Stopping auf Validierung ab, `y[val_idx]` im Loss); "" = nicht maskierbar (ETTh1:
    # Validierung ist ein Serienstueck ohne getrennte Ziele, Training darauf ist kein Label-Leck)
    mask_val_cmd: str = ""
    # B2 (v5): Kantenrichtungs-Probe auf einem synthetischen Graphen im Format des Datensatzes: (Pfad der npz im Branch, Knoten, Val, Test)
    edge_probe: tuple[str, int, int, int] | None = None
    probe_dims: tuple[int, int] = (7, 1433)   # (Klassen, Merkmale) des synthetischen Probe-Graphen, je Datensatz
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


# Planetoid-Datensaetze (Standard-Split, Kipf & Welling): eine Task-Fabrik, parametrisiert ueber PLANETOID (harness/planetoid_data.py SPECS).
# name: (Anzeigename, Knoten, Merkmale, Klassen, Trainingsknoten, Merkmalsart, Zusatzhinweis)
PLANETOID = {
    "cora": ("Cora", 2708, 1433, 7, 140, "bag-of-words (binary, not normalised)", ""),
    "citeseer": ("CiteSeer", 3327, 3703, 6, 120, "bag-of-words (binary, not normalised)",
                 "15 isolated nodes have an all-zero feature row and no edges; none of them is in train_idx, val_idx or test_idx.\n"),
    "pubmed": ("PubMed", 19717, 500, 3, 60, "TF-IDF weights (non-negative floats, not normalised)",
               "Large graph (19717 nodes): do not build dense N x N matrices (about 1.5 GB each), use sparse operations.\n"),
}
PLANETOID_VAL, PLANETOID_TEST = 500, 1000


def planetoid_data_description(name: str) -> str:
    label, n, f, c, n_train, kind, extra = PLANETOID[name]
    return f"""\
Dataset: {label} citation graph (Planetoid split), file /data/{name}_public.npz (numpy.load):
- features: float32 [{n}, {f}] {kind}
- edges: int64 [E, 2] directed pairs (u, v), already contains both directions
- labels: int64 [{n}], class 0..{c - 1}; -1 for all test nodes (labels hidden)
- train_idx ({n_train}), val_idx ({PLANETOID_VAL}), test_idx ({PLANETOID_TEST}): node indices
{extra}Task: node classification. Train on train_idx (use val_idx for model selection if you want),
predict a class for every node in test_idx.
Output: write /output/predictions.csv with header `id,prediction`, one row per test node,
id = node index from test_idx, prediction = integer class. Nothing else is scored.
Environment: Python 3.11, CPU only, numpy, scipy, torch preinstalled. Read the random seed from
the env var SEED (int) and seed numpy and torch with it. Must finish in < 5 minutes."""


CORA_DATA_DESCRIPTION = planetoid_data_description("cora")


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


def planetoid_task(name: str) -> Task:
    _, n_nodes, n_features, n_classes, _, _, _ = PLANETOID[name]
    path = f"/data/{name}_public.npz"
    prep = [
        _file_cmd(HARNESS_DIR / "planetoid_data.py", "/opt/prep/planetoid_data.py"),
        _file_cmd(HARNESS_DIR / f"{name}_data.py", f"/opt/prep/{name}_data.py"),
        "pip install -q numpy scipy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        f"python /opt/prep/{name}_data.py && rm -rf /opt/prep",
        data_check_cmd(path),
    ]
    return Task(
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["planetoid_data.py", f"{name}_data.py", "metrics.py", "verify.py", "planetoid_eval.py", f"{name}_eval.py"],
        eval_cmd=f"python {{harness_dir}}/{name}_eval.py",
        data_description=planetoid_data_description(name),
        code_notes=(
            "Implement the model yourself in PyTorch (no torch_geometric available). "
            f"Print the final validation accuracy to stdout. Use only {path}. "
            f"{CORA_SMOKE}"
        ),
        ablation_cmd=(
            f"python -c \"import numpy as np; d = dict(np.load('{path}')); "
            "d['edges'] = np.zeros((0, 2), dtype=np.int64); "
            f"np.savez_compressed('{path}', **d)\""
        ),
        edge_probe=(path, n_nodes, PLANETOID_VAL, PLANETOID_TEST),
        probe_dims=(n_classes, n_features),
        mask_val_cmd=(
            f"python -c \"import numpy as np; d = dict(np.load('{path}')); "
            "v = d['val_idx']; l = d['labels'].copy(); l[v] = np.random.RandomState(0).permutation(l[v]); d['labels'] = l; "
            f"np.savez_compressed('{path}', **d)\""
        ),
        smoke_ids_cmd=(   # erwartete ids = die ersten N (aufsteigend) von test_idx, aus dem Datensatz selbst
            f"python -c \"import numpy as np; d = np.load('{path}'); "
            f"print(*sorted(int(i) for i in d['test_idx'])[:{CORA_SMOKE_N}])\""
        ),
        val_notes=CORA_VAL_NOTES,
        val_harness_files=("planetoid_val_eval.py", f"{name}_val_eval.py"),
        val_eval_cmd=f"python {{harness_dir}}/{name}_val_eval.py",
        val_smoke_ids_cmd=(
            f"python -c \"import numpy as np; d = np.load('{path}'); "
            f"print(*sorted(int(i) for i in d['val_idx'])[:{CORA_SMOKE_N}])\""
        ),
    )


def cora_task() -> Task:
    return planetoid_task("cora")


def citeseer_task() -> Task:
    return planetoid_task("citeseer")


def pubmed_task() -> Task:
    return planetoid_task("pubmed")


# ETT (Horizont 96): eine Task-Fabrik, parametrisiert ueber ETT (harness/ett_data.py SPECS).
# name: (Anzeigename, Takt, Einheit eines Schritts, Zeilen Train, Zeilen Val, Testfenster, Val-Fenster, Val-Schrittweite)
ETT = {
    "etth1": ("ETTh1", "hourly", "hours", 8640, 2880, 2785, 117, 24),
    "etth2": ("ETTh2", "hourly", "hours", 8640, 2880, 2785, 117, 24),
    "ettm1": ("ETTm1", "15-minute", "15-minute steps", 34560, 11520, 11425, 120, 96),
    "ettm2": ("ETTm2", "15-minute", "15-minute steps", 34560, 11520, 11425, 120, 96),
}
ETTH1_VAL_WINDOWS = 117   # = etth1_data.n_val_windows(96): Schrittweite 24 in den 2880 Val-Zeilen
ETTH1_SMOKE_WINDOWS = 8


def ett_data_description(name: str) -> str:
    label, gran, unit, n_train, n_val, n_win, _, _ = ETT[name]
    kind = "hourly" if gran == "hourly" else "15-minute"
    return f"""\
Dataset: {label} (electricity transformer temperature, {gran}), 7 channels, multivariate forecasting.
File /data/{name}_public.npz (numpy.load), all values already z-score standardised with the
mean/std of the training rows:
- train: float32 [{n_train}, 7] (first 12 months)
- val: float32 [{n_val}, 7] (next 4 months)
- test_inputs: float32 [{n_win}, 720, 7]; test_inputs[i] is the 720 {kind} steps just before the
  forecast start of test window i (window i forecasts {unit} i .. i+95 of the 4-month test period).
Task: for every test window i in 0..{n_win - 1} forecast the next 96 {unit} of all 7 channels. You may
use only the last L steps of each test_inputs[i] (L <= 720 is your choice).
Never use one window's input to predict another window (each forecast must depend only on its
own history).
Output: write /output/predictions.csv with header `id,prediction`, one row per value:
id = "{{window}}_{{step}}_{{channel}}" (window 0..{n_win - 1}, step 0..95, channel 0..6), prediction = float in
standardised units. {n_win}*96*7 = {n_win * 96 * 7} rows. Nothing else is scored (MSE and MAE over all values).
Environment: Python 3.11, CPU only, numpy and torch preinstalled. Read the random seed from the
env var SEED (int) and seed numpy and torch with it. Must finish in < 5 minutes."""


ETTH1_DATA_DESCRIPTION = ett_data_description("etth1")


def ett_val_notes(name: str) -> str:
    _, _, unit, n_train, _, _, n_val_win, stride = ETT[name]
    return (
        "Also write /output/val_predictions.csv (header `id,prediction`) for validation windows, same id format "
        f"'{{window}}_{{step}}_{{channel}}', window 0..{n_val_win - 1}, step 0..95, channel 0..6 ({n_val_win}*96*7 rows): validation window w "
        f"forecasts the 96 {unit} starting at row {n_train} + {stride}*w of the series train followed by val (i.e. val rows {stride}*w .. {stride}*w+95). "
        "Its input is only the history before that start, taken from train followed by val (never later rows). "
        "In smoke mode (SMOKE=1) write only validation windows 0..7. "
        "Selection among candidate scripts uses only this file; it is never scored as the result."
    )


ETTH1_VAL_NOTES = ett_val_notes("etth1")


def ett_smoke(name: str) -> str:
    n_win = ETT[name][5]
    return (
        "Smoke mode: if env var SMOKE=1, use at most 1 training batch of 32 samples for 1 epoch, predict only "
        f"test windows 0..{ETTH1_SMOKE_WINDOWS - 1} and write predictions.csv with only these windows "
        f"({ETTH1_SMOKE_WINDOWS}*96*7 rows); it must finish in under 60 seconds. "
        "The pipeline runs this first to check the output shape (windows, 96, 7). If SMOKE is unset (or not 1) the script "
        f"must do the normal full run: all {n_win} windows, full training. Never treat an unset SMOKE as smoke mode."
    )


ETTH1_SMOKE = ett_smoke("etth1")


def ett_task(name: str) -> Task:
    path = f"/data/{name}_public.npz"
    prep = [
        _file_cmd(HARNESS_DIR / "ett_data.py", "/opt/prep/ett_data.py"),
        _file_cmd(HARNESS_DIR / f"{name}_data.py", f"/opt/prep/{name}_data.py"),
        "pip install -q numpy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        f"python /opt/prep/{name}_data.py && rm -rf /opt/prep",
        data_check_cmd(path),
    ]
    smoke_ids = tuple(f"{w}_{s}_{c}" for w in range(ETTH1_SMOKE_WINDOWS) for s in range(96) for c in range(7))
    return Task(
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["ett_data.py", f"{name}_data.py", "metrics.py", "verify.py", "diagnostics.py", "droplast.py", "ett_eval.py", f"{name}_eval.py"],
        eval_cmd=f"python {{harness_dir}}/{name}_eval.py",
        data_description=ett_data_description(name),
        code_notes=(
            "Implement the model yourself in PyTorch. Write predictions.csv efficiently (numpy "
            "to text, not per-row Python objects). Print the final validation MSE to stdout. "
            f"Use only {path}. "
            f"{ett_smoke(name)}"
        ),
        smoke_ids=smoke_ids,
        smoke_dims=(ETTH1_SMOKE_WINDOWS, 96, 7),
        smoke_dim_names=("windows", "steps", "channels"),
        val_notes=ett_val_notes(name),
        val_harness_files=("ett_val_eval.py", f"{name}_val_eval.py"),
        val_eval_cmd=f"python {{harness_dir}}/{name}_val_eval.py",
        val_smoke_ids=smoke_ids,
        val_smoke_dims=(ETTH1_SMOKE_WINDOWS, 96, 7),
    )


def etth1_task() -> Task:
    return ett_task("etth1")


def etth2_task() -> Task:
    return ett_task("etth2")


def ettm1_task() -> Task:
    return ett_task("ettm1")


def ettm2_task() -> Task:
    return ett_task("ettm2")


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
        harness_files=["planetoid_data.py", "cora_data.py", "cora_lcc_data.py", "metrics.py", "verify.py", "cora_lcc_eval.py"],
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
        prepare_files=("planetoid_data.py", "cora_data.py", "cora_lcc_data.py"),
        prepare_cmd="mkdir -p /data && python {dir}/cora_lcc_data.py public {split} /data/cora_lcc_public.npz",
        seeds_per_split=CORA_LCC_SEEDS_PER_SPLIT,
    )


TASKS = {"Cora": cora_task, "ETTh1": etth1_task, "Cora-LCC": cora_lcc_task}

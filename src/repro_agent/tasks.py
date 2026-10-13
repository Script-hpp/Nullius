"""Datensatz-spezifische Aufgaben: Setup-Kommandos, Eval-Kommando, Datenbeschreibung fuer `code`.

Der Orchestrator kennt keine Datensaetze, nur `Task`. Neue Papers = neuer Eintrag hier.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import re
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
    # Einwand-Experimente (objection_experiments.py): Einstellungen, die ein Einwand aendern darf (Schluessel, Beschreibung).
    # Daten, Splits und Metrik gehoeren nie dazu; leer = keine parameter_change-Experimente fuer diesen Task.
    tunable: tuple[tuple[str, str], ...] = ()
    # Protokoll-Neubewertungen (protocol_change): Namen der Re-Evaluationen, die der Harness aus denselben Vorhersagen rechnet
    # (kein neuer Lauf). "drop_last" = Metrik ueber die Testfenster, die ein Loader mit drop_last=True liefert (harness/droplast.py).
    protocols: tuple[str, ...] = ()
    # Ausgabe-Helfer (harness/nullius_io.py, Flag `io_helper` des Orchestrators): Konstanten als JSON, Text des Ausgabeabschnitts
    io_config: str = ""             # JSON fuer CONFIG des Helfers; "" = Task ohne Helfer
    io_output: str = ""             # ersetzt den Abschnitt "Output: ..." der Datenbeschreibung
    io_smoke: str = ""              # ersetzt die Smoke-Anweisung in code_notes
    io_val_notes: str = ""          # ersetzt val_notes
    selection_contract: bool = False

    def selection_info(self) -> dict | None:
        if not self.selection_contract:
            return None
        cfg = json.loads(self.io_config)
        info = {'version': 'withheld_validation_v1', 'heldout_fraction': .2, 'kind': cfg['kind'],
                'checkpoint_validation': 'visible validation only', 'train_test_split': 'unchanged'}
        if cfg['kind'] == 'windows':
            name = Path(cfg['data']).stem.removesuffix('_public')
            first = math.ceil(int(ETT[name][4] * .8) / cfg['val_stride'])
            info.update(selection_samples=cfg['val_windows']-first, total_validation_samples=cfg['val_windows'],
                        inference='causal_stream_v2', deterministic_rechecks=8,
                        calendar_features=['year','month','day','weekday','hour'] + (['minute'] if name.startswith('ettm') else []))
        else:
            total = 210 if 'cora_lcc' in cfg['data'] else 500
            info.update(selection_samples=math.ceil(total*.2), total_validation_samples=total)
        return info

    def with_independent_selection(self) -> "Task":
        if self.selection_contract or not self.io_config:
            return self
        cfg = json.loads(self.io_config)
        cfg['selection_contract'] = True
        notes = (' Independent candidate selection: the final 20% of validation observations are withheld from fitting. '
                 'The visible validation observations remain available for checkpoint choice. Original training and test splits are unchanged. ')
        changes = dict(selection_contract=True, io_config=json.dumps(cfg),
                       harness_files=[*self.harness_files, 'selection_contract.py'],
                       val_harness_files=(*self.val_harness_files, 'selection_contract.py'),
                       val_eval_cmd='NULLIUS_SELECTION_HOLDOUT=1 ' + self.val_eval_cmd)
        pins = {'numpy': '2.4.6', 'scipy': '1.17.1', 'torch': '2.14.1', 'coverage': '7.16.2'}
        changes['setup_cmds'] = [re.sub(r'\b(numpy|scipy|torch|coverage)\b(?=\s|$)',
                                      lambda m: m.group(1) + '==' + pins[m.group(1)], cmd)
                                 if cmd.startswith('pip install ') else cmd for cmd in self.setup_cmds]
        if cfg['kind'] == 'nodes':
            changes.update(data_description=self.data_description + notes +
                           'selection_idx names the hidden selection nodes; their labels are -1. val_idx contains only visible checkpoint nodes.',
                           val_notes='Write val_predictions.csv for selection_idx, not val_idx. ' + notes,
                           io_val_notes='nullius_io writes selection predictions from the whole-graph output. ' + notes,
                           val_smoke_ids_cmd=self.val_smoke_ids_cmd.replace("['val_idx']", "['selection_idx']"))
        else:
            changes['setup_cmds'] = [('NULLIUS_FORECAST_CALENDAR=1 ' + cmd) if cmd.startswith('python /opt/prep/')
                                     else cmd for cmd in changes['setup_cmds']]
            cfg['val_stride'] = ETT[Path(cfg['data']).stem.removesuffix('_public')][7]
            changes['io_config'] = json.dumps(cfg)
            forecast_output = (
                'Define train_forecaster(data, seed, smoke) -> predictor(history). No training or file writing at module import. '
                'The factory fits only on data[\"train\"], uses the visible data[\"val\"] for checkpoint selection, seeds numpy and torch, '
                'and returns a callable that predicts [96,7] from one numpy history [720,7]. Crop each history to the paper input length. '
                'Use nullius_io.forecast_data(length: int, stride: int = 1) -> dict for exact train/val input and target arrays; '
                'these have REAL targets, never zeros. It returns a dictionary with named keys, not a tuple. '
                'Its first argument is the integer history length, not a split name; it accepts no split argument. '
                'Use this API example (length is the paper input length):\n```python\n'
                'windows = nio.forecast_data(length)\n'
                'train_x = windows["train_inputs"]\ntrain_y = windows["train_targets"]\n'
                'val_x = windows["val_inputs"]\nval_y = windows["val_targets"]\n'
                'train_calendar = windows.get("train_calendar")\nval_calendar = windows.get("val_calendar")\n```\n'
                'Input shapes are [N,length,7]; target shapes are [N,96,7]. '
                'The same helper supplies train_calendar and val_calendar aligned with each input window. '
                'If the method uses timestamps, return predictor(history, *, calendar): calendar contains the same past '
                '720 rows, with columns year, month, day, weekday (Monday=0), hour, plus minute for ETTm. '
                'Crop calendar together with history. A predictor(history) without calendar remains valid for methods that do not use dates. '
                'Keep the trained model in eval mode and use deterministic inference without gradient updates or fitting on supplied histories. '
                'No full test_inputs is exposed to the factory. Our isolated runner supplies one history only after the preceding forecast is recorded. '
                'Store temporary checkpoints under /tmp/forecast-agent. /output belongs to the runner and is not writable. '
                'Do not write these files yourself. The runner writes predictions.csv and val_predictions.csv.'
            )
            desc = re.sub(r'- test_inputs:.*?Output:.*?\nEnvironment:',
                          lambda _: 'Task: multivariate forecasting, horizon 96, input length <=720.\nOutput: ' + forecast_output + '\nEnvironment:',
                          self.data_description, count=1, flags=re.S)
            changes.update(data_description=desc + notes, io_output=forecast_output, code_notes=
                           'Implement the paper model in PyTorch. ' + forecast_output +
                           ' Smoke mode: smoke=True requires one batch and one epoch; the runner checks eight windows. ' + notes,
                           io_smoke='Smoke mode: nio.SMOKE is True when smoke=True. Fit one batch for one epoch. The runner checks eight windows.',
                           val_notes=notes, io_val_notes=notes)
        return dataclasses.replace(self, **changes)

    def env_for(self, seed: int) -> str:
        if self.seeds_per_split:
            return f"SPLIT={seed // self.seeds_per_split} SEED={seed % self.seeds_per_split}"
        return f"SEED={seed}"

    def split_for(self, seed: int) -> int:
        return seed // self.seeds_per_split if self.seeds_per_split else 0

    def io_helper_source(self) -> str:
        """Quelltext des Ausgabe-Helfers fuer diesen Task (nullius_io.py mit eingesetzten Konstanten)."""
        src = (HARNESS_DIR / "nullius_io.py").read_text()
        line = "CONFIG: dict = {}  # __CONFIG__"
        assert line in src and self.io_config
        cfg = dict(json.loads(self.io_config), val=bool(self.val_eval_cmd))
        return src.replace(line, f"CONFIG = {cfg!r}", 1)

    def with_io_helper(self) -> "Task":
        """Dieselbe Aufgabe mit dem Ausgabevertrag ueber den Helfer: Output-Abschnitt, Smoke-Anweisung und Val-Hinweis ersetzt.
        Gilt fuer beide Arme gleich; Inhalt der Aufgabe (Daten, Metrik, Toleranz) bleibt unveraendert."""
        if not self.io_config:
            return self
        desc, n = re.subn(r"Output:.*?\nEnvironment:", lambda _: f"Output: {self.io_output}\nEnvironment:", self.data_description, count=1, flags=re.S)
        assert n == 1, "Output-Abschnitt der Datenbeschreibung nicht gefunden"
        notes = self.code_notes
        smoke_start = notes.index("Smoke mode:")
        notes = notes[:smoke_start] + self.io_smoke
        return dataclasses.replace(self, data_description=desc, code_notes=notes, val_notes=self.io_val_notes if self.val_notes else "")


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


CORA_TUNABLE = (
    ("hidden_units", "number of hidden units per layer"),
    ("dropout", "dropout rate"),
    ("weight_decay", "L2 regularisation / weight decay"),
    ("learning_rate", "optimiser learning rate"),
    ("epochs", "number of training epochs"),
    ("num_layers", "number of layers"),
    ("feature_normalization", "row-normalisation of the node features"),
)
ETTH1_TUNABLE = (
    ("input_window", "number of past hourly steps used as input (L <= 720)"),
    ("epochs", "number of training epochs"),
    ("learning_rate", "optimiser learning rate"),
    ("batch_size", "batch size"),
    ("individual", "one linear layer per channel (individual) versus one shared layer"),
    ("decomposition_kernel", "moving-average kernel size of the trend/seasonal decomposition"),
)


IO_NODES_HELP = (
    "use the helper module nullius_io (the file nullius_io.py lies next to train.py): `import nullius_io as nio`; `d = nio.load()` returns "
    "the arrays of the data file (same keys as numpy.load); at the end call `nio.write_predictions(pred)` with the predicted class of EVERY "
    "node of the graph (integer array of shape [{n}] indexable by node id, or logits of shape [{n}, {c}]; numpy array or torch tensor). "
    "The helper writes /output/predictions.csv (header `id,prediction`, one row per test node, id = node index from test_idx) and "
    "/output/val_predictions.csv itself. Do not write these files yourself. Nothing else is scored."
)
IO_NODES_SMOKE = (
    "Smoke mode: nio.SMOKE is True when env var SMOKE=1. Then train exactly 1 epoch; it must finish in under 60 seconds. "
    "nio.write_predictions applies the smoke subset of nodes itself: call it exactly as in the full run, with predictions for all nodes. "
    "The pipeline runs this first to check the output form. If SMOKE is unset (or not 1) the script must do the normal full run: full "
    "training. Never treat an unset SMOKE as smoke mode."
)
IO_NODES_VAL = (
    "The same call writes /output/val_predictions.csv from the same predictions (same trained model, same run): pass the predictions for "
    "all nodes and the helper selects val_idx. Selection among candidate scripts uses only this file; it is never scored as the result."
)


def planetoid_task(name: str) -> Task:
    _, n_nodes, n_features, n_classes, _, _, _ = PLANETOID[name]
    path = f"/data/{name}_public.npz"
    prep = [
        _file_cmd(HARNESS_DIR / "planetoid_data.py", "/opt/prep/planetoid_data.py"),
        _file_cmd(HARNESS_DIR / "data_identity.py", "/opt/prep/data_identity.py"),
        _file_cmd(HARNESS_DIR / f"{name}_data.py", f"/opt/prep/{name}_data.py"),
        "pip install -q numpy scipy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        f"python /opt/prep/{name}_data.py && rm -rf /opt/prep",
        data_check_cmd(path),
    ]
    return Task(
        tunable=CORA_TUNABLE,
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["data_identity.py", "planetoid_data.py", f"{name}_data.py", "metrics.py", "verify.py", "planetoid_eval.py", f"{name}_eval.py"],
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
        io_config=json.dumps({"kind": "nodes", "data": path, "smoke_n": CORA_SMOKE_N}),
        io_output=IO_NODES_HELP.format(n=n_nodes, c=n_classes),
        io_smoke=IO_NODES_SMOKE,
        io_val_notes=IO_NODES_VAL,
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


def ett_io_output(name: str) -> str:
    n_win, n_val_win = ETT[name][5], ETT[name][6]
    return (
        "use the helper module nullius_io (the file nullius_io.py lies next to train.py): `import nullius_io as nio`; `d = nio.load()` "
        "returns the arrays of the data file (same keys as numpy.load); at the end call `nio.write_predictions(test_pred, val_pred)` with "
        f"test_pred = standardised forecasts of ALL {n_win} test windows, shape [{n_win}, 96, 7], and val_pred = forecasts of the "
        f"{n_val_win} validation windows, shape [{n_val_win}, 96, 7] (numpy arrays or torch tensors). The helper writes "
        "/output/predictions.csv (header `id,prediction`, id = \"{window}_{step}_{channel}\") and /output/val_predictions.csv itself. "
        "Do not write these files yourself. Nothing else is scored (MSE and MAE over all values)."
    )


def ett_io_smoke(name: str) -> str:
    n_win = ETT[name][5]
    return (
        "Smoke mode: nio.SMOKE is True when env var SMOKE=1. Then use at most 1 training batch of 32 samples for 1 epoch and pass "
        f"forecasts for at least the first {ETTH1_SMOKE_WINDOWS} test windows and the first {ETTH1_SMOKE_WINDOWS} validation windows to "
        f"nio.write_predictions (passing all windows also works, the helper takes the first {ETTH1_SMOKE_WINDOWS}); it must finish in "
        "under 60 seconds. The pipeline runs this first to check the output form. If SMOKE is unset (or not 1) the script must do the "
        f"normal full run: all {n_win} windows, full training. Never treat an unset SMOKE as smoke mode."
    )


def ett_io_val_notes(name: str) -> str:
    _, _, unit, n_train, _, _, n_val_win, stride = ETT[name]
    return (
        f"Validation windows for val_pred: window w (0..{n_val_win - 1}) forecasts the 96 {unit} starting at row {n_train} + {stride}*w of "
        f"the series train followed by val (i.e. val rows {stride}*w .. {stride}*w+95). Its input is only the history before that start, "
        "taken from train followed by val (never later rows). Selection among candidate scripts uses only val_pred; it is never scored as "
        "the result."
    )


def ett_task(name: str) -> Task:
    path = f"/data/{name}_public.npz"
    prep = [
        _file_cmd(HARNESS_DIR / "ett_data.py", "/opt/prep/ett_data.py"),
        _file_cmd(HARNESS_DIR / "data_identity.py", "/opt/prep/data_identity.py"),
        _file_cmd(HARNESS_DIR / f"{name}_data.py", f"/opt/prep/{name}_data.py"),
        "pip install -q numpy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu",
        f"python /opt/prep/{name}_data.py && rm -rf /opt/prep",
        data_check_cmd(path),
    ]
    smoke_ids = tuple(f"{w}_{s}_{c}" for w in range(ETTH1_SMOKE_WINDOWS) for s in range(96) for c in range(7))
    return Task(
        tunable=ETTH1_TUNABLE,
        protocols=("drop_last",),
        image="python:3.11-slim",
        setup_cmds=prep,
        harness_files=["data_identity.py", "ett_data.py", f"{name}_data.py", "metrics.py", "verify.py", "diagnostics.py", "droplast.py", "ett_eval.py", f"{name}_eval.py"],
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
        io_config=json.dumps({"kind": "windows", "data": path, "steps": 96, "channels": 7, "test_windows": ETT[name][5],
                              "val_windows": ETT[name][6], "smoke_windows": ETTH1_SMOKE_WINDOWS}),
        io_output=ett_io_output(name),
        io_smoke=ett_io_smoke(name),
        io_val_notes=ett_io_val_notes(name),
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
        tunable=CORA_TUNABLE,
        image=base.image,
        setup_cmds=["pip install -q numpy scipy torch coverage --extra-index-url https://download.pytorch.org/whl/cpu"],
        harness_files=["data_identity.py", "planetoid_data.py", "cora_data.py", "cora_lcc_data.py", "metrics.py", "verify.py", "cora_lcc_eval.py"],
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
        prepare_files=("data_identity.py", "planetoid_data.py", "cora_data.py", "cora_lcc_data.py"),
        prepare_cmd="mkdir -p /data && python {dir}/cora_lcc_data.py public {split} /data/cora_lcc_public.npz",
        seeds_per_split=CORA_LCC_SEEDS_PER_SPLIT,
        io_config=json.dumps({"kind": "nodes", "data": "/data/cora_lcc_public.npz", "smoke_n": CORA_SMOKE_N}),
        io_output=IO_NODES_HELP.format(n=2485, c=7),
        io_smoke=IO_NODES_SMOKE,
        io_val_notes=IO_NODES_VAL,
    )


# Freigegeben nach benchmark/experiments/results/datasets_v2_validation.md: CiteSeer. Nicht freigegeben (Referenz verfehlt Regel): PubMed, ETTh2, ETTm1, ETTm2.
TASKS = {"Cora": cora_task, "CiteSeer": citeseer_task, "ETTh1": etth1_task, "Cora-LCC": cora_lcc_task}

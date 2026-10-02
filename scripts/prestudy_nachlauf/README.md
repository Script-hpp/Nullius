# Vorversuch "Nachlauf", Schritt 1 (lokal, Handarbeit)

Original-Code der Autoren (RTSF/RLinear, LTSF-Linear/DLinear) lokal auf CPU, Vorhersagen ueber einen Adapter, Auswertung mit dem unveraenderten Harness.
Vorab-Festlegung: `docs/prestudy_nachlauf.md`. Ergebnis: `benchmark/experiments/results/nachlauf_prestudy.md`. Kein LLM, kein Tavily, keine Sandbox, kein `src/`-Eingriff.

Arbeitsverzeichnis ausserhalb des Repos: `W=/home/onuralp/Desktop/hackathon/prestudy_work` (Klone `rtsf/`, `ltsf/`, `data/ETTh1.csv`, `venv/`, `ckpt/`, `out/`).

## Aufbau
1. Umgebung: `uv venv --python 3.11 $W/venv; uv pip install numpy pandas scikit-learn matplotlib torch --extra-index-url https://download.pytorch.org/whl/cpu; uv pip install "numpy<2"` (numpy 2 scheitert, siehe Ergebnis).
2. Klone auf festem Commit: `git clone https://github.com/plumprc/RTSF $W/rtsf && git -C $W/rtsf checkout 0fec00104f754f4fbf795b9b4da5fa2459b32e76`; `git clone https://github.com/cure-lab/LTSF-Linear $W/ltsf && git -C $W/ltsf checkout 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`.
3. Daten: `curl -L -o $W/data/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`, Kopie nach `rtsf/dataset/` und `ltsf/dataset/` (beide Orte sind in den `.gitignore` der Klone).

## Startbefehle (cwd = Klon, venv aktiv)
- RTSF, Repo-Seed 1024, unveraendertes `run.py`:
  `python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model RLinear --rev --seq_len 336 --pred_len 96 --gpu 3`
  (`script.md:3` schreibt `--model Linear`: das scheitert mit `KeyError: 'Linear'`, Rohausgabe `nachlauf_prestudy_rlinear_literal.txt`.)
- LTSF, Repo-Seed 2021, unveraendertes `run_longExp.py` (Zeile Horizont 96 aus `etth1.sh`):
  `python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.005`
- CPU: beide Repos setzen `use_gpu = torch.cuda.is_available() and args.use_gpu`; ohne CUDA faellt der Code selbst auf CPU (`Use CPU`). `--use_gpu` (`type=bool`) wird nicht angefasst, `--gpu 3` bleibt wirkungslos.

## Adapter und Auswertung (Skripte hier)
- `common.py`: Zuordnungspruefung (`check_mapping`), `predictions.csv` schreiben, unveraenderten Harness `etth1_eval.py` per `runpy` ausfuehren (nur der feste Pfad `/output/predictions.csv` wird auf die lokale Datei umgebogen).
- `adapter_rtsf.py`, `adapter_ltsf.py`: Checkpoint laden, Test-Loader mit `drop_last=False`, `shuffle=False`, Zuordnungspruefung, danach Vorhersagen.
  `python adapter_rtsf.py --repo $W/rtsf --ckpt <checkpoint.pth> --csv $W/data/ETTh1.csv --out <dir>`
- `evaluate.py <predictions.csv>`: JSON-Zeile des Harness (MSE alle Fenster, `metrics_droplast`).
- Seeds 0 bis 3: `seed_copies/*_run_seed<N>.py` sind Kopien des Startskripts, nur `fix_seed = ...` ist anders (`diff` im Ergebnis). Sie liegen ausserhalb des Klons, `PYTHONPATH` zeigt auf den Klon.
  `run_rtsf_seed.sh <N>` und `run_ltsf_seed.sh <N>` (Training, Checkpoint sichern, Adapter, Harness; Ausgaben nach `$W/out/`).
- `summarize.py $W`: Tabelle ueber alle Seeds.
- Der Lauf mit dem Repo-Seed (1024 bzw. 2021) wurde mit den Startbefehlen oben direkt gestartet; Checkpoint danach nach `$W/ckpt/<einheit>_seed<Seed>/` kopiert, Adapter und `evaluate.py` wie oben.

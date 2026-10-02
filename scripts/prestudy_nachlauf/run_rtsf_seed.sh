#!/usr/bin/env bash
# Startbefehl aus RTSF script.md:3 (ETTh1, 96) mit Seed-Kopie von run.py; nur `--model RLinear` statt `Linear` (siehe README), sonst identisch.
# Aufruf: run_rtsf_seed.sh <seed>   (Umgebung: venv aktiv). Klon bleibt unveraendert (Kopie liegt ausserhalb, PYTHONPATH auf den Klon).
set -euo pipefail
W=/home/onuralp/Desktop/hackathon/prestudy_work
S=$(cd "$(dirname "$0")" && pwd)
seed=$1
cd $W/rtsf
rm -rf checkpoints
{ time PYTHONPATH=$W/rtsf python -u $S/seed_copies/rtsf_run_seed$seed.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model RLinear --rev --seq_len 336 --pred_len 96 --gpu 3 ; } > $W/out/rtsf_seed$seed.txt 2>&1
mkdir -p $W/ckpt/rtsf_seed$seed && cp checkpoints/*/checkpoint.pth $W/ckpt/rtsf_seed$seed/
python $S/adapter_rtsf.py --repo $W/rtsf --ckpt $W/ckpt/rtsf_seed$seed/checkpoint.pth --csv $W/data/ETTh1.csv --out $W/out/rtsf_seed$seed
python $S/evaluate.py $W/out/rtsf_seed$seed/predictions.csv > $W/out/rtsf_seed$seed/harness.json

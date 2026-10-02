#!/usr/bin/env bash
# Zeile fuer Horizont 96 aus LTSF-Linear scripts/EXP-LongForecasting/Linear/etth1.sh (seq_len=336, DLinear) mit Seed-Kopie von run_longExp.py.
# Aufruf: run_ltsf_seed.sh <seed>   (Umgebung: venv aktiv). Klon bleibt unveraendert.
set -euo pipefail
W=/home/onuralp/Desktop/hackathon/prestudy_work
S=$(cd "$(dirname "$0")" && pwd)
seed=$1
cd $W/ltsf
rm -rf checkpoints
{ time PYTHONPATH=$W/ltsf python -u $S/seed_copies/ltsf_run_seed$seed.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.005 ; } > $W/out/ltsf_seed$seed.txt 2>&1
mkdir -p $W/ckpt/ltsf_seed$seed && cp checkpoints/*/checkpoint.pth $W/ckpt/ltsf_seed$seed/
python $S/adapter_ltsf.py --repo $W/ltsf --ckpt $W/ckpt/ltsf_seed$seed/checkpoint.pth --csv $W/data/ETTh1.csv --out $W/out/ltsf_seed$seed
python $S/evaluate.py $W/out/ltsf_seed$seed/predictions.csv > $W/out/ltsf_seed$seed/harness.json

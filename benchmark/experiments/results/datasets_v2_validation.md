# Datasets v2: validation results (complete, 2026-10-07)

Protocol: `benchmark/experiments/datasets_v2_validation.md` (fixed before the first run). Runner: `scripts/run_reference_local.py <unit> 5`
(local CPU, 16 cores, torch 2.14 CPU, references from `tests/reference/`, data prepared by the harness code, labels loaded fresh by the harness code).
Seeds 0..4. Runtime per seed = wall time of the reference script on a shared 16-core CPU (other work ran at the same time, numbers are upper bounds).
A one-seed pipeline probe of GCN CiteSeer (0.6970, 40 s) ran before the formal runs; the formal run below repeats seeds 0..4.

Status: complete. Released: CiteSeer only. Not released: PubMed (GAT fails), ETTh2, ETTm1, ETTm2 (DLinear fails). Rule applied as pre-registered (every reference per dataset
within +-1 pp accuracy / +-0.01 MSE of the paper value, 5 seeds), no tuning. PubMed GAT: lr corrected from 0.005 to 0.01 by the protocol addendum of 2026-10-07
(transcription error of the protocol, the paper text states 0.01 for PubMed); the first run (lr 0.005) is kept below as the failed first run.

## Raw output (copied from the run logs)

### gcn_citeseer
```
seed 0: accuracy 0.6970 (39s) epochs 200 val acc 0.7060
seed 1: accuracy 0.7120 (38s) epochs 196 val acc 0.7140
seed 2: accuracy 0.7080 (45s) epochs 200 val acc 0.7120
seed 3: accuracy 0.7240 (61s) epochs 200 val acc 0.7100
seed 4: accuracy 0.7090 (57s) epochs 185 val acc 0.7060
gcn_citeseer: 5/5 gueltig; accuracy Mittel 0.7100, Std 0.0097; Ziel 0.703 +- 0.01, Abstand 0.0070 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 48, max 61; Einzelwerte [0.697, 0.712, 0.708, 0.724, 0.709]
```

### sgc_citeseer
```
seed 0: accuracy 0.7190 (26s) weight decay 0.0001 val acc 0.7380
seed 1: accuracy 0.7210 (4s) weight decay 0.0001 val acc 0.7380
seed 2: accuracy 0.7190 (5s) weight decay 0.0001 val acc 0.7380
seed 3: accuracy 0.7190 (2s) weight decay 0.0001 val acc 0.7380
seed 4: accuracy 0.7220 (4s) weight decay 5e-05 val acc 0.7360
sgc_citeseer: 5/5 gueltig; accuracy Mittel 0.7200, Std 0.0014; Ziel 0.719 +- 0.01, Abstand 0.0010 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 8, max 26; Einzelwerte [0.719, 0.721, 0.719, 0.719, 0.722]
```

### gat_citeseer
```
seed 0: accuracy 0.7100 (83s) epochs 358 best val acc 0.7260 seconds 82
seed 1: accuracy 0.7160 (117s) epochs 625 best val acc 0.7300 seconds 116
seed 2: accuracy 0.7110 (107s) epochs 554 best val acc 0.7300 seconds 106
seed 3: accuracy 0.7330 (99s) epochs 467 best val acc 0.7400 seconds 99
seed 4: accuracy 0.7150 (152s) epochs 661 best val acc 0.7320 seconds 151
gat_citeseer: 5/5 gueltig; accuracy Mittel 0.7170, Std 0.0093; Ziel 0.725 +- 0.01, Abstand 0.0080 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 112, max 152; Einzelwerte [0.71, 0.716, 0.711, 0.733, 0.715]
```

### gcn_pubmed
```
seed 0: accuracy 0.7910 (72s) epochs 154 val acc 0.7960
seed 1: accuracy 0.7910 (81s) epochs 179 val acc 0.7880
seed 2: accuracy 0.7920 (57s) epochs 122 val acc 0.7900
seed 3: accuracy 0.7920 (62s) epochs 125 val acc 0.7960
seed 4: accuracy 0.7900 (58s) epochs 129 val acc 0.7900
gcn_pubmed: 5/5 gueltig; accuracy Mittel 0.7912, Std 0.0008; Ziel 0.79 +- 0.01, Abstand 0.0012 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 66, max 81; Einzelwerte [0.791, 0.791, 0.792, 0.792, 0.79]
```

### sgc_pubmed
```
seed 0: accuracy 0.7880 (4s) weight decay 5e-05 val acc 0.8160
seed 1: accuracy 0.7880 (3s) weight decay 5e-05 val acc 0.8160
seed 2: accuracy 0.7880 (5s) weight decay 0.0001 val acc 0.8180
seed 3: accuracy 0.7860 (4s) weight decay 0.0001 val acc 0.8160
seed 4: accuracy 0.7880 (4s) weight decay 0.0001 val acc 0.8160
sgc_pubmed: 5/5 gueltig; accuracy Mittel 0.7876, Std 0.0009; Ziel 0.789 +- 0.01, Abstand 0.0014 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 4, max 5; Einzelwerte [0.788, 0.788, 0.788, 0.786, 0.788]
```

### gat_pubmed, first run (lr 0.005, interrupted after 2 of 5 seeds, superseded by the corrected protocol)
```
seed 0: accuracy 0.7610 (145s) epochs 388 best val acc 0.8100 seconds 144
seed 1: accuracy 0.7610 (128s) epochs 325 best val acc 0.8140 seconds 127
```

### gat_pubmed, rerun with lr 0.01 (protocol addendum 2026-10-07), counts
```
seed 0: accuracy 0.7640 (282s) epochs 386 best val acc 0.8160 seconds 281
seed 1: accuracy 0.7590 (148s) epochs 188 best val acc 0.8120 seconds 147
seed 2: accuracy 0.7610 (301s) epochs 398 best val acc 0.8100 seconds 300
seed 3: accuracy 0.7530 (72s) epochs 166 best val acc 0.8180 seconds 72
seed 4: accuracy 0.7490 (70s) epochs 177 best val acc 0.8200 seconds 69
gat_pubmed: 5/5 gueltig; accuracy Mittel 0.7572, Std 0.0061; Ziel 0.79 +- 0.01, Abstand 0.0328 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 175, max 301; Einzelwerte [0.764, 0.759, 0.761, 0.753, 0.749]
gat_pubmed: 5/5 gueltig; accuracy Mittel 0.7572, Std 0.0061; Ziel 0.79 +- 0.01, Abstand 0.0328 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 175, max 301; Einzelwerte [0.764, 0.759, 0.761, 0.753, 0.749]
```

### dlinear_etth2
```
seed 0: mse 0.3364 (6s) val_mse 0.22785495221614838
seed 1: mse 0.3146 (6s) val_mse 0.226071298122406
seed 2: mse 0.3323 (4s) val_mse 0.24164903163909912
seed 3: mse 0.3611 (4s) val_mse 0.2457917183637619
seed 4: mse 0.3314 (4s) val_mse 0.24231122434139252
dlinear_etth2: 5/5 gueltig; mse Mittel 0.3352, Std 0.0167; Ziel 0.289 +- 0.01, Abstand 0.0462 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 5, max 6; Einzelwerte [0.3364, 0.3146, 0.3323, 0.3611, 0.3314]
dlinear_etth2: 5/5 gueltig; mse Mittel 0.3352, Std 0.0167; Ziel 0.289 +- 0.01, Abstand 0.0462 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 5, max 6; Einzelwerte [0.3364, 0.3146, 0.3323, 0.3611, 0.3314]
```

### nlinear_etth2
```
seed 0: mse 0.2839 (5s) val_mse 0.20660582184791565
seed 1: mse 0.2778 (4s) val_mse 0.20760077238082886
seed 2: mse 0.2759 (4s) val_mse 0.20823778212070465
seed 3: mse 0.2865 (3s) val_mse 0.20967262983322144
seed 4: mse 0.2786 (4s) val_mse 0.20788833498954773
nlinear_etth2: 5/5 gueltig; mse Mittel 0.2805, Std 0.0045; Ziel 0.277 +- 0.01, Abstand 0.0035 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 4, max 5; Einzelwerte [0.2839, 0.2778, 0.2759, 0.2865, 0.2786]
nlinear_etth2: 5/5 gueltig; mse Mittel 0.2805, Std 0.0045; Ziel 0.277 +- 0.01, Abstand 0.0035 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 4, max 5; Einzelwerte [0.2839, 0.2778, 0.2759, 0.2865, 0.2786]
```

### dlinear_ettm1
```
seed 0: mse 0.3357 (28s) val_mse 0.4118722081184387
seed 1: mse 0.3365 (17s) val_mse 0.40701207518577576
seed 2: mse 0.3420 (21s) val_mse 0.41428685188293457
seed 3: mse 0.3333 (19s) val_mse 0.41880589723587036
seed 4: mse 0.3456 (36s) val_mse 0.41820770502090454
dlinear_ettm1: 5/5 gueltig; mse Mittel 0.3386, Std 0.0050; Ziel 0.299 +- 0.01, Abstand 0.0396 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 24, max 36; Einzelwerte [0.3357, 0.3365, 0.342, 0.3333, 0.3456]
dlinear_ettm1: 5/5 gueltig; mse Mittel 0.3386, Std 0.0050; Ziel 0.299 +- 0.01, Abstand 0.0396 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 24, max 36; Einzelwerte [0.3357, 0.3365, 0.342, 0.3333, 0.3456]
```

### nlinear_ettm1
```
seed 0: mse 0.3081 (13s) val_mse 0.38445860147476196
seed 1: mse 0.3021 (13s) val_mse 0.38726332783699036
seed 2: mse 0.3024 (23s) val_mse 0.3873484432697296
seed 3: mse 0.3004 (29s) val_mse 0.38817062973976135
seed 4: mse 0.3147 (18s) val_mse 0.3825075328350067
nlinear_ettm1: 5/5 gueltig; mse Mittel 0.3055, Std 0.0059; Ziel 0.306 +- 0.01, Abstand 0.0005 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 19, max 29; Einzelwerte [0.3081, 0.3021, 0.3024, 0.3004, 0.3147]
nlinear_ettm1: 5/5 gueltig; mse Mittel 0.3055, Std 0.0059; Ziel 0.306 +- 0.01, Abstand 0.0005 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 19, max 29; Einzelwerte [0.3081, 0.3021, 0.3024, 0.3004, 0.3147]
```

### dlinear_ettm2
```
seed 0: mse 0.2023 (14s) val_mse 0.13889475166797638
seed 1: mse 0.1997 (21s) val_mse 0.13954822719097137
seed 2: mse 0.2021 (35s) val_mse 0.13843463361263275
seed 3: mse 0.2525 (29s) val_mse 0.16693218052387238
seed 4: mse 0.2136 (31s) val_mse 0.1434183418750763
dlinear_ettm2: 5/5 gueltig; mse Mittel 0.2140, Std 0.0221; Ziel 0.167 +- 0.01, Abstand 0.0470 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 26, max 35; Einzelwerte [0.2023, 0.1997, 0.2021, 0.2525, 0.2136]
dlinear_ettm2: 5/5 gueltig; mse Mittel 0.2140, Std 0.0221; Ziel 0.167 +- 0.01, Abstand 0.0470 -> Referenzurteil nicht reproduziert
Laufzeit je Seed (s): Mittel 26, max 35; Einzelwerte [0.2023, 0.1997, 0.2021, 0.2525, 0.2136]
```

### nlinear_ettm2
```
seed 0: mse 0.1649 (21s) val_mse 0.11650162935256958
seed 1: mse 0.1653 (16s) val_mse 0.11662948131561279
seed 2: mse 0.1645 (22s) val_mse 0.11647256463766098
seed 3: mse 0.1647 (17s) val_mse 0.11649613827466965
seed 4: mse 0.1650 (17s) val_mse 0.11653290688991547
nlinear_ettm2: 5/5 gueltig; mse Mittel 0.1649, Std 0.0003; Ziel 0.167 +- 0.01, Abstand 0.0021 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 18, max 22; Einzelwerte [0.1649, 0.1653, 0.1645, 0.1647, 0.165]
nlinear_ettm2: 5/5 gueltig; mse Mittel 0.1649, Std 0.0003; Ziel 0.167 +- 0.01, Abstand 0.0021 -> Referenzurteil reproduziert
Laufzeit je Seed (s): Mittel 18, max 22; Einzelwerte [0.1649, 0.1653, 0.1645, 0.1647, 0.165]
```

## Reading

| Dataset | Reference | Mean +- std (5 seeds) | Paper value (quote in protocol file) | Distance | Rule +-1 pp / +-0.01 | Runtime per seed |
|---|---|---|---|---|---|---|
| CiteSeer | GCN | 0.7100 +- 0.0097 | 0.703 (Kipf Table 2) | 0.0070 | pass | 48 s mean, 61 s max |
| CiteSeer | SGC | 0.7200 +- 0.0014 | 0.719 (Wu Table 2, "Our experiments") | 0.0010 | pass | 8 s, 26 s max |
| CiteSeer | GAT | 0.7170 +- 0.0093 | 0.725 (Velickovic Table 2) | 0.0080 | pass | 112 s, 152 s max |
| PubMed | GCN | 0.7912 +- 0.0008 | 0.790 (Kipf Table 2) | 0.0012 | pass | 66 s, 81 s max |
| PubMed | SGC | 0.7876 +- 0.0009 | 0.789 (Wu Table 2) | 0.0014 | pass | 4 s, 5 s max |
| PubMed | GAT (lr 0.01) | 0.7572 +- 0.0061 | 0.790 (Velickovic Table 2) | 0.0328 | FAIL | 175 s mean, 301 s max |
| ETTh2 | DLinear | 0.3352 +- 0.0167 | 0.289 (Zeng Table 2) | 0.0462 | FAIL | 5 s, 6 s max |
| ETTh2 | NLinear | 0.2805 +- 0.0045 | 0.277 | 0.0035 | pass | 4 s, 5 s max |
| ETTm1 | DLinear | 0.3386 +- 0.0050 | 0.299 | 0.0396 | FAIL | 24 s, 36 s max |
| ETTm1 | NLinear | 0.3055 +- 0.0059 | 0.306 | 0.0005 | pass | 19 s, 29 s max |
| ETTm2 | DLinear | 0.2140 +- 0.0221 | 0.167 | 0.0470 | FAIL | 26 s, 35 s max |
| ETTm2 | NLinear | 0.1649 +- 0.0003 | 0.167 | 0.0021 | pass | 18 s, 22 s max |

## Decision (rule applied, 2026-10-07)

| Dataset | Released | Reason |
|---|---|---|
| CiteSeer | yes | GCN, SGC, GAT pass |
| PubMed | no | GAT fails (0.7572 vs 0.790). GCN and SGC pass on the same data, so the harness is not the cause. |
| ETTh2 | no | DLinear fails (0.3352 vs 0.289), NLinear passes |
| ETTm1 | no | DLinear fails (0.3386 vs 0.299), NLinear passes |
| ETTm2 | no | DLinear fails (0.2140 vs 0.167), NLinear passes |

The coordinator confirmed the strict rule (2026-10-07): no protocol change, no change of a reference.

## Open points (not investigated, no tuning)

- PubMed GAT: best validation accuracy 0.81-0.82 while test accuracy is 0.75-0.76 in both runs (lr 0.005 and lr 0.01). Hypothesis only: the reference differs from the original
  PubMed setup in a detail not in the stored paper text. Not verified.
- DLinear on ETTh2, ETTm1, ETTm2: NLinear passes on all three, DLinear fails on all three, in the same direction (MSE too high by 0.04-0.05). The pattern points to the reference
  settings (DLinear is run unchanged from ETTh1), not to the harness. Hypothesis only. A follow-up with per-dataset settings from the original scripts would be post-hoc
  and a separate task, pending the user's decision.
- ETTm risk (7.7M prediction rows per seed = 11425 windows x 96 x 7): the references, which write the full CSV and pass the harness check (load, plausibility, metric), take 18-36 s per seed in total
  on a shared 16-core CPU. The test suite measures writing a 2785-window array (1.87M rows, 38.7 MB) at 1.18 s, so the full ETTm array (about 159 MB) takes about 5 s. No problem for the live demo budget
  on the local CPU; not measured in the Nebius sandbox (ETTm is not released, so no sandbox run was made).

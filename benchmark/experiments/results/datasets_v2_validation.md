# Datasets v2: validation results (PARTIAL, interrupted 2026-10-05 00:38 local)

Protocol: `benchmark/experiments/datasets_v2_validation.md` (fixed before the first run). Runner: `scripts/run_reference_local.py <unit> 5`
(local CPU, 16 cores, torch 2.14 CPU, references from `tests/reference/`, data prepared by the harness code, labels loaded fresh by the harness code).
Seeds 0..4. Runtime per seed = wall time of the reference script on a shared 16-core CPU (other work ran at the same time, numbers are upper bounds).
A one-seed pipeline probe of GCN CiteSeer (0.6970, 40 s) ran before the formal runs; the formal run below repeats seeds 0..4.

Status: CiteSeer all three references PASS. PubMed GCN and SGC PASS. PubMed GAT interrupted after 2 of 5 seeds, both far below target (see below).
ETTh2, ETTm1, ETTm2 (DLinear, NLinear) not run yet.

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

### gat_pubmed
```
seed 0: accuracy 0.7610 (145s) epochs 388 best val acc 0.8100 seconds 144
seed 1: accuracy 0.7610 (128s) epochs 325 best val acc 0.8140 seconds 127
```

## Reading

| Dataset | Reference | Mean +- std (5 seeds) | Paper value (quote in protocol file) | Distance | Rule +-0.01 | Runtime per seed |
|---|---|---|---|---|---|---|
| CiteSeer | GCN | 0.7100 +- 0.0097 | 0.703 (Kipf Table 2) | 0.0070 | pass | 48 s mean, 61 s max |
| CiteSeer | SGC | 0.7200 +- 0.0014 | 0.719 (Wu Table 2, "Our experiments") | 0.0010 | pass | 8 s, 26 s max |
| CiteSeer | GAT | 0.7170 +- 0.0093 | 0.725 (Velickovic Table 2) | 0.0080 | pass | 112 s, 152 s max |
| PubMed | GCN | 0.7912 +- 0.0008 | 0.790 (Kipf Table 2) | 0.0012 | pass | 66 s, 81 s max |
| PubMed | SGC | 0.7876 +- 0.0009 | 0.789 (Wu Table 2) | 0.0014 | pass | 4 s, 5 s max |
| PubMed | GAT | 2 of 5 seeds: 0.761, 0.761 | 0.790 (Velickovic Table 2) | about 0.029 | would FAIL if it holds | about 130-145 s |

PubMed GAT is not released as is. Not yet investigated: the reference uses lr 0.005 (paper text states only 8 output heads and L2 0.001 for PubMed;
the original code uses its own PubMed settings, not in the stored fixtures). Best val acc 0.81 vs test 0.761 suggests a tuning gap in the reference, not a harness
bug (GCN and SGC on the same PubMed data pass). Next: finish the 5 seeds as pre-registered, then investigate (lr 0.01 is the usual PubMed value in the original
repo, but it is not in the stored texts: ask the coordinator before using it) and report. Do not retune silently.

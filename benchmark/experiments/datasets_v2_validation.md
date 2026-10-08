# Datasets v2: validation protocol (fixed BEFORE the first reference run)

Date fixed: 2026-10-04. Scope: new tasks CiteSeer, PubMed (Planetoid split), ETTh2, ETTm1, ETTm2 (horizon 96).
No LLM call, no Tavily call. Not a benchmark claim: product reach only. Benchmark stays v4.

## Rule

A dataset is released for discovery, catalog and UI only if every reference solution below,
run through the own harness (data prepared by the harness code, labels/targets loaded fresh by the
harness, metric from `harness/metrics.py`), has

- mean over >= 5 seeds (SEED=0..4) within **+-1 percentage point** (accuracy) or **+-0.01 MSE** (ETT) of the paper value.

Otherwise: dataset not released, cause investigated and reported. Changes to a reference or to the harness
after a miss are allowed only for documented harness bugs and are listed with the first (failed) run.
Runtime per seed on CPU is recorded. PubMed must run in minutes per seed, else reported.

## Paper values (quoted from the stored paper texts in `benchmark/fixtures/`)

| Dataset | Reference | Target | Source quote |
|---|---|---|---|
| CiteSeer | GCN (Kipf) | 0.703 | 1609.02907.json, Table 2: "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s) 66 .0 (48s)" (columns Citeseer, Cora, Pubmed, NELL) |
| PubMed | GCN (Kipf) | 0.790 | same row |
| CiteSeer | SGC (Wu) | 0.719 | 1902.07153.json, Table 2 "Our experiments": "SGC 81 .0 ± 0.0 71 .9 ± 0.1 78 .9 ± 0.0" (Cora, Citeseer, Pubmed) |
| PubMed | SGC (Wu) | 0.789 | same row |
| CiteSeer | GAT (Velickovic) | 0.725 | 1710.10903.json, Table 2: "GAT (ours) 83.0 ± 0.7% 72.5 ± 0.7% 79.0 ± 0.3%" (Cora, Citeseer, Pubmed) |
| PubMed | GAT (Velickovic) | 0.790 | same row |
| ETTh2 | DLinear / NLinear | 0.289 / 0.277 | 2205.13504.json, Table 2, ETTh2 horizon 96: "0.288 0.352 0.277 0.338 0.289 0.353" = Linear, NLinear, DLinear (MSE, MAE) |
| ETTm1 | DLinear / NLinear | 0.299 / 0.306 | same table, ETTm1 96: "0.308 0.352 0.306 0.348 0.299 0.343" |
| ETTm2 | DLinear / NLinear | 0.167 / 0.167 | same table, ETTm2 96: "0.168 0.262 0.167 0.255 0.167 0.260" |

Column order in that table: IMP., Linear*, NLinear*, DLinear*, FEDformer, ... (header "Methods IMP. Linear* NLinear* DLinear* FEDformer ...").
Note: Table 1 of the paper labels ETTm1/ETTm2 as "5min" with 69,680 steps; the ETDataset files are 15-minute data
(69,680 rows = 725 days x 96). The split protocol below is the one of the public code (Autoformer/DLinear data loader).

## Protocol of the new harness data

- Planetoid: raw files from `kimiyoung/planetoid`; standard split: train = first 20 per class (Cora 140, CiteSeer 120, PubMed 60),
  val = next 500, test = 1000 (indices from `test.index`). CiteSeer isolated test nodes are zero-padded as in Kipf's `utils.py`.
- GAT on PubMed uses the paper's changed setup (8 output heads averaged, L2 0.001, "we have applied K = 8 output attention heads
  (instead of one), and strengthened the L2 regularization to λ = 0 .001"); learning rate stays 0.005 (not stated for PubMed in the paper text).
- ETT: hourly (ETTh2) 12/4/4 months = rows [0, 8640) / [8640, 11520) / [11520, 14400); 15-minute (ETTm1, ETTm2) x4 =
  [0, 34560) / [34560, 46080) / [46080, 57600). z-score with train statistics, 7 channels, test windows stride 1,
  horizon 96, history up to 720 steps given to the agent. Reference models DLinear and NLinear run UNCHANGED from ETTh1 (L = 336).
  (Original per-dataset learning rates may differ; a miss will be investigated and reported, not silently tuned.)

## Addendum 2026-10-07: correction of a transcription error (PubMed GAT learning rate)

The protocol line above ("learning rate stays 0.005 (not stated for PubMed in the paper text)") is wrong. It is a transcription error of this protocol,
not a change of the rule. The stored paper text `benchmark/fixtures/1710.10903.json` states: "with an initial learning rate of 0.01 for Pubmed,
and 0.005 for all other datasets". The reference for GAT on PubMed therefore uses lr 0.01 (`scripts/run_reference_local.py`, `patch_planetoid`).
Everything else (8 output heads, L2 0.001, tolerance, seeds, rule) stays as pre-registered; nothing was tuned.
The first run with lr 0.005 (2 of 5 seeds, both 0.761) stays in the results file as the failed first run. The rerun with lr 0.01 (5 seeds) is the run that counts.
The user approved this correction.

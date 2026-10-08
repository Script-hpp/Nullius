# Resume notes: new datasets (CiteSeer, PubMed, ETTh2, ETTm1, ETTm2)

Branch `datasets-v2` (from `main`), task from the Orca coordinator (task_f41d0287a82a). Paused 2026-10-05 by coordinator request. No push, no merge.
Forbidden: every LLM call, every Tavily call, `curate_catalog.py` without `--dry-run`. Do not write `benchmark/fixtures/` (check `git status benchmark/` after every test run:
the suite creates `benchmark/fixtures/repos/search_2511.08094.json`, delete it, never commit it).

## Done (committed)
1. Protocol pre-registered: `benchmark/experiments/datasets_v2_validation.md` (rule, paper quotes, targets).
2. Planetoid harness parametrized: `harness/planetoid_data.py` (+ thin `cora_data.py`, `citeseer_data.py`, `pubmed_data.py`), `planetoid_eval.py`, `planetoid_val_eval.py`.
   Cora output byte-identical (hash test `tests/test_planetoid.py`, opt-in real data: `PLANETOID_NETWORK_TESTS=1`). `tasks.py`: `planetoid_task(name)`, `citeseer_task`, `pubmed_task`.
   `edge_direction.make_probe` takes class/feature counts, `Task.probe_dims`; `code_check` blocks `ind.(cora|citeseer|pubmed)`.
3. ETT harness parametrized: `harness/ett_data.py` (+ thin `etth1/etth2/ettm1/ettm2_data.py`), `ett_eval.py`, `ett_val_eval.py`; `tasks.py`: `ett_task(name)`, `etth2_task`, `ettm1_task`, `ettm2_task`.
   ETTh1 output identical (hash test `tests/test_ett.py`, `ETT_NETWORK_TESTS=1`). ETTm: 15-minute data, borders x4 (34560/46080/57600), val stride 96 (120 val windows), 11425 test windows.
   `tests/test_droplast.py` adapted to `ett_eval`.
4. `scripts/run_reference_local.py` supports units `gcn|sgc|gat_{citeseer,pubmed}` and `dlinear|nlinear_{etth2,ettm1,ettm2}`.

## Done (WIP commit, not verified)
- Step 4 edits (NOT registered in `TASKS`, so nothing is live): `discovery.py` (ETT_DATASETS, ALLOWED_METRICS, queries, hints, keywords), `catalog.py` (ETT_TASKS, TASK_SEEDS/METRICS/SUFFIX, `_task_of`),
  `paper_head.py`, web (`glossary.ts`, `discovery.ts`, `labels.ts`, `Discovery.tsx`, `Catalog.tsx`, new `web/src/datasets.test.ts`). Web build/lint/test and python suite NOT yet run on these edits.

## Validation status (`benchmark/experiments/results/datasets_v2_validation.md`)
- PASS: CiteSeer GCN 0.7100, SGC 0.7200, GAT 0.7170; PubMed GCN 0.7912, SGC 0.7876.
- PubMed GAT: 2 of 5 seeds 0.761 (target 0.790): likely fails. Needs investigation, see result file.
- Pending: ETTh2, ETTm1, ETTm2 with DLinear and NLinear (targets in protocol file); PubMed GAT seeds 2-4.

## Commands to continue
Environment (torch is not in the project env): `uv venv --python 3.11 $S/venv && uv pip install --python $S/venv/bin/python numpy scipy torch pytest -e . --extra-index-url https://download.pytorch.org/whl/cpu`
(`$S` = scratchpad dir, any directory works).
```
PYTHONPATH=src $S/venv/bin/python scripts/run_reference_local.py gat_pubmed 5 --cache $S/refcache --out $S/out/gat_pubmed.txt
for u in dlinear_etth2 nlinear_etth2 dlinear_ettm1 nlinear_ettm1 dlinear_ettm2 nlinear_ettm2; do
  PYTHONPATH=src $S/venv/bin/python scripts/run_reference_local.py $u 5 --cache $S/refcache --out $S/out/$u.txt; done
```
Append raw output and the table rows to `benchmark/experiments/results/datasets_v2_validation.md`. Run units one after another (runtime numbers).
Then: register only passed datasets in `tasks.TASKS` (keys `CiteSeer`, `PubMed`, `ETTh2`, `ETTm1`, `ETTm2`) and trim discovery/catalog/web to the released set;
add Fake tests for registered tasks; `uv run python -m pytest -m "not integration" -q` (needs `uv sync` with torch not required); `cd web && npm ci && npm run build && npm run lint && npm test`.
ETTm risk: 7.7M prediction rows per seed (11425*96*7), check agent-side CSV writing time and harness memory/time (Nebius sandbox validation of one ETTm reference through `Orchestrator._run_and_eval`, see `scripts/validate_etth1_harness.py`).
Final report: commit hashes; per dataset released yes/no, mean +- std vs paper value with quote source, runtime per seed; changed files; README sentence; licenses (Planetoid data: kimiyoung/planetoid repo, check its license
file before stating; ETDataset: CC BY-ND 4.0 per its README, verify); open points.

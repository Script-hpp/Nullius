# Pre-registration: Tavily findings become experiments (role 3, stage 2)

Written and committed **before** the first run (2026-10-07). Rules below are not changed after seeing results; results that contradict the hope are reported. Design and rules: `docs/tavily_experiments.md`. Code: `src/repro_agent/objection_experiments.py`.

## Question

Does the pipeline turn verified Tavily findings into correct, traceable experiments on real papers, without touching the main verdict? This is a **function test of the feature**, not a benchmark: no accuracy number, no comparison between arms, no claim that Tavily improves reproduction accuracy (CLAUDE.md: no further Tavily accuracy claims before submission). Nothing from this test enters the benchmark v4 results or protocols.

## Fixed rules (copied from the design note, approved 2026-10-07)

- Types: `seed_variance`, `parameter_change`, `reported_number`, `not_measurable`; the LLM chooses, fixed code demotes to `not_measurable` when the value is not in the quote, the setting is not in the task's list (`Task.tunable`), the metric is not the claim's, or a seed objection names no seed/variation word.
- `seed_variance`: K extra seeds (5; quick mode 3) with the final main code, own branches. `bestaetigt` if at least 2 valid extra seeds and (`std(main+extra) >= tolerance` or `|mean(extra) - main mean| > tolerance`); `widerlegt` if neither and at least `ceil(K/2)` valid; else `offen`.
- `parameter_change`: code role changes exactly one setting (value must appear in the new code), 3 paired seeds (the first three seeds of the main run), `SE = sqrt((s_new^2 + s_main^2)/2)/sqrt(n)`. `bestaetigt` if `|delta| > max(tolerance, 2*SE)`; `widerlegt` if `|delta| <= tolerance` and at least 2 valid pairs; else `offen`.
- `reported_number`: no run. `bestaetigt` / `widerlegt` only if the quote names the same setup (`same_setup == "yes"`), by `|ours - reported| <= tol` and `|reported - target| > tol` / `|ours - target| <= tol` and `|ours - reported| > tol`; else `offen`.
- Limits: at most 3 experiments per run (quick mode 2), duplicates run once, run budget `MAX_COST_PER_RUN_USD` stops experiments.
- The main verdict, `dossier.attempts` and the harness contract are unchanged by the feature.

## Part 1: fixture dress rehearsal on DLinear (real LLM, real sandbox, no Tavily)

Command: `LLM_PROVIDER=tokenfactory uv run python scripts/run_objection_fixture.py --paper zeng_dlinear_2023` (script committed with this file). Claim `dlinear_etth1_96_mse` (target 0.375, tolerance 0.01, 5 seeds), roles 2/4 from stored fixtures, community findings from `benchmark/fixtures/role3_zeng_objections.json` (one stored Tavily call, 0 credits), full mode (not quick), one run. Cost counts against the 1.00 USD gross cap.

Success criteria, all required (otherwise "not met"):

1. The run completes and writes `dossier.json` with `objection_checks`, `events.jsonl` with the four new event types, and `zeugnis.md` with the section "Objections checked".
2. Every entry keeps a verbatim quote that occurs in a page stored in the fixture and the URL of that page.
3. At least 1 entry has a type other than `not_measurable` **and** the status of a `seed_variance` or `parameter_change` entry rests on at least 2 valid measurements taken in own branches (branch ids in `attempts`, none of them in `dossier.attempts`).
4. For every entry, the status follows the fixed rule applied to the stored numbers (checked by recomputing from `measured`/`attempts` in the evaluation, no manual judgement).
5. The main verdict (`urteil`, mean, std) of the claim is computed from main attempts only (no objection attempt in it).

What I expect, not a success criterion: the "input window 336 vs 96" objection becomes a `parameter_change` (the claim text and plan already use 336, so a `widerlegt` or `offen` is likely, not `bestaetigt`); the fairness objection becomes `not_measurable`; the issue #117 style number (`mse 0.426, not in paper 0.386`) becomes `reported_number` with `same_setup = "unknown"` and status `offen`. A model that classifies #117 as `yes` would be an error to report. If the verified findings of this run do not contain a measurable objection (criterion 3), the criterion is not met and that is the result.

## Part 2: live runs (after approval by the coordinator)

Two live runs, one per paper, `LLM_PROVIDER=tokenfactory`, `TAVILY_MODE=live`, `sandbox=nebius`, product configuration (`graph_repair=True`, `objection_experiments=True`), **quick mode off** (full seeds) unless the coordinator decides otherwise:

- Kipf & Welling 2017 GCN (`kipf_gcn_2017`, Cora, claim `gcn_cora_acc`, 10 seeds). Role 3 needs a live search (no fixture exists).
- Zeng 2023 DLinear (`zeng_dlinear_2023`, ETTh1, claim `dlinear_etth1_96_mse`, 5 seeds).

**Hard caps** for parts 1 and 2 together: Token Factory **1.00 USD gross**, Tavily **60 credits** (counted with `src/repro_agent/tavily_usage.py` from the start timestamp of part 2; the estimator rounds up). Crawl always with `limit` (the community round uses 3). Stop rule: if the running totals pass 90 % of a cap before a run starts, that run does not start. No re-runs to improve a result; a failed run is reported as failed and not repeated, except for a clearly infrastructural failure (sandbox setup), which may be repeated once if the caps allow.

Success criteria per live run, all required:

1. The run completes with a verdict, and `objection_checks` is not empty, or the log shows why (the community search found no verified finding; reported as such, not as a pass).
2. Criteria 2, 4, 5 of part 1 hold.
3. Added wall time of the objection round (`objection_round_end.seconds`) and its cost (`cost_net_usd`) are recorded in the result.

Acceptance of the whole task (coordinator): at least one real exported run (`scripts/export_site_data.py`, production runs only) whose Evidence view shows a checked objection ("Objections checked" block with quote, URL, measurement, status).

## Report (phase D)

`benchmark/experiments/results/tavily_experiments_test.md`: every criterion with its numbers from the files (dossier, events, `logs/costs.jsonl`, `logs/tavily_usage.jsonl`), "met / not met", costs and credits against the caps, counts per type and status, what did not work. No claim of improved accuracy.

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

## Addendum A (2026-10-08): post hoc query pivot and one more DLinear run

Written and committed **before** the run described here. This is a **post hoc pivot after coordinator web research**: parts 1 and 2 (four runs) yielded no measurable live objection, and the coordinator then searched the web by hand (not with Tavily) and found objections outside the channels our role-3 queries cover: the LTSF-Linear test loader sets `drop_last=True` and so drops part of the test set (documented in the FITS README and the SparseTSF paper); discussions of GAT/SGC on discuss.dgl.ai; the Shchur 2018 split critique for GCN. This addendum changes the **search**, not the checks. It does not touch the rules above.

### What changes (generic, same for every paper; no paper-specific term, no hard-coded URL)

1. Query templates (`tavily_tools.CRITIQUE_TEMPLATES`), method and dataset are the only substitutions, one Tavily search each (1 credit each, 5 results):
   - `<method> <dataset> reproduce results differ`
   - `<method> bug evaluation`
   - `<method> baseline re-evaluation fixed`
   - `we re-run <method> on <dataset> and compare with the reported numbers` (follow-up papers and repos)
   Domains: `github.com`, `arxiv.org`, `openreview.net` kept; discussion forums added for the first three (`discuss.dgl.ai`, `discuss.pytorch.org`, `discuss.huggingface.co`, `stats.stackexchange.com`, `reddit.com`). The old three searches stay.
2. Issue bodies: a numbered issue/PR/discussion hit whose text is shorter than 1,500 characters (a heading, not a post) is re-read with one Extract call (at most 4 pages). If no numbered issue page of the paper's repo was found at all (only the `/issues` list page, or nothing), the existing crawl of the repo's issues runs (limit 3, unchanged); it returns issue bodies.
3. Unchanged: crawl `limit`, the quote check (the quote must occur in the page), `names_paper` (a finding must name the method, the full title or the arXiv ID; repo pages exempt), noise filter, page-size filter, objection classification and all rules above. Community cache version raised to 3 so older entries are not reused (`--fresh-community` is used anyway).
4. New objection type `protocol_change` (only for tasks that declare it; ETTh1 declares `drop_last`). The classifier may choose it only if the quote says that the test loader drops the last incomplete batch (fixed code demotes otherwise: keyword check on statement and quote, claim metric must be mse/mae, task must support it). **No new branch and no agent code.** Our harness already computes, for every valid main seed, the claim metric over the test windows a `drop_last=True` loader would keep (`harness/droplast.py`, batch sizes 32/64/128/256, from the same predictions as the full value, `Attempt.metrics_droplast`); the type only evaluates these stored numbers. The agent never sees or computes them. Rule (fixed now): `effect = |truncated mean - full mean|` per batch size (or only the size named in the quote), over the same seeds. `bestaetigt` if at least 2 valid seeds and the largest effect >= tolerance / 4 (the same 1/4 threshold the existing drop_last report note uses); `widerlegt` if at least 2 valid seeds and all effects are below it; `offen` otherwise. The Evidence line states the number of windows kept and the nearest-to-target batch size. A `bestaetigt` says: re-scoring the same predictions on a drop_last test set moves the metric by at least tolerance / 4. It does not say that the paper used drop_last.

### Run (phase D of this addendum)

One live run: `zeng_dlinear_2023`, claim `dlinear_etth1_96_mse`, 5 seeds, full mode, `--fresh-community`, `LLM_PROVIDER=tokenfactory`, `TAVILY_MODE=live`, sandbox `nebius`, product configuration with `objection_experiments=True`. Caps for the whole test (unchanged except Tavily): Token Factory **1.00 USD gross** (0.733 used, a run does not start above 0.90; ask first if the run would exceed it), Tavily **80 credits** (raised by the coordinator from 60; about 41 used, a run does not start above 72). No re-run to improve a result; an infrastructure failure may be repeated once only with the coordinator's approval.

### Honest expectation (not a success criterion)

The `drop_last` finding for DLinear is hoped for, but may not appear: the follow-up sources name `drop_last` in READMEs and an appendix, and our queries and the page filters may still not return them, or the extractor quote may not pass `names_paper` (it often says "LTSF-Linear" or "the original code", not "DLinear"). If it appears and passes, I expect `widerlegt` more likely than `bestaetigt`: for 2785 test windows the stored drop_last effect on ETTh1 DLinear-type models is about 0.001 to 0.004 MSE (see `droplast_note.py`), i.e. close to the 0.0025 threshold. A result in either direction is reported as is. If no `protocol_change` entry appears, the result is "not found" and is reported with the hits and drop reasons from the log (they are logged since commit 25c4f82).

### Acceptance (unchanged)

The criteria of parts 1 and 2 stand as written. Criterion 3 asks for a status resting on 2 valid measurements **in own branches**; a `protocol_change` rests on re-scored main-branch predictions, so by construction it does not satisfy that wording. It is reported separately ("met" is not claimed for it under criterion 3). The coordinator's acceptance (an exported run whose Evidence view shows a checked objection with quote, URL, measurement, status) is judged on the result as it stands.

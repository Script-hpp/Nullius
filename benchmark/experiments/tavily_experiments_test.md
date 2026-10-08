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

## Addendum B (2026-10-08): post hoc page-level `names_paper` for setup findings, and one more DLinear run

Written and committed **before** the run described here. This is a **post hoc rule change, labelled as such**: it was chosen after seeing the result of Addendum A, and the user approved it explicitly on 2026-10-08. It changes one filter, not the checks that keep a finding honest.

### Why (result of Addendum A, `results/tavily_experiments_addendum_a.md`)

The new queries found the FITS README. An offline replay of the extractor on that README returned the `drop_last` statement with a verbatim quote, but `names_paper` dropped it: the statement and quote do not name DLinear (the page lists DLinear among affected models elsewhere). The live run kept 0 findings. The blocker was the statement-level filter, not the search.

### Rule change (generic, no paper-specific term)

1. A candidate whose topic is evaluation, protocol or setup (the measurable classes: protocol, parameter, seed; decided by a keyword check on topic, statement and quote, `community.SETUP_KEYWORDS`: the protocol keywords plus seed, batch size, learning rate, input length, lookback, window size, data split, evaluation protocol) passes `names_paper` if the **page text** names the method (whole word), the full title or the arXiv ID. Reason logged per finding: `pass_reason = "page_names_method"` (otherwise `statement_names_method`; repo pages stay exempt).
2. Candidates of any other topic keep the statement-level rule unchanged.
3. The quote check stays strict: the quote must occur verbatim on the page. No other check is touched (page-size filter, noise filter, objection classification, experiment rules, tolerance/4 threshold).
4. Known limit, stated now: pages are read only if they already name the title, the arXiv ID, or (method-search hits) method plus dataset. So for a non-repo page that was read, the page-level test is almost always true. The rule therefore means in practice "a setup finding on a page about the method counts", which is weaker than before. A page-level mention does not show that the statement is about the method (the FAN example in Addendum A, a label-leak statement about another repo's code, would pass if its topic were a setup topic and the page named DLinear). I accept this for one run and report any such finding as it is, with the quote.
5. Raw extractor candidates are now stored per page (`community.candidates`, also in the `community` event): url, topic, statement, quote, `quote_in_page`, and `outcome` (`kept_page_names_method`, `kept_statement_names_method`, `dropped_quote_not_in_page`, `dropped_names_neither_method_title_nor_arxiv_id`, `dropped_duplicate`). Earlier runs stored only drops, which is why the FITS case could not be explained.

Tests (offline): setup finding passes by page; dropped when the page does not name the method; non-setup finding keeps the statement rule; quote not on page stays dropped; candidates stored.

### Run

One live run: `zeng_dlinear_2023`, claim `dlinear_etth1_96_mse`, 5 seeds, full mode, `--fresh-community`, `LLM_PROVIDER=tokenfactory`, `TAVILY_MODE=live`, sandbox `nebius`, `objection_experiments=True`. Caps for the whole test (raised by the user): Token Factory **1.25 USD gross** (about 0.92 used; the run starts only if it fits with margin, a run cost about 0.18), Tavily **80 credits** (about 59 used; stop at 78, a run used about 18, so it fits only if the community round plus roles use at most about 19). No re-run to improve a result.

### Honest expectation (not a success criterion)

I expect the FITS `drop_last` finding to pass now if Tavily returns the README again and the extractor repeats it (not deterministic: it did not appear in the live run of Addendum A). If it passes, the `protocol_change` check reads the stored drop_last numbers; I expect `widerlegt` more likely than `bestaetigt` (effect near the 0.0025 threshold). Any other outcome is reported as is. The main verdict (about 0.39, `teilweise`) is unrelated.

### Acceptance (unchanged)

As in Addendum A. A `protocol_change` entry rests on re-scored main-branch predictions, so criterion 3 (2 measurements in own branches) is not claimed for it.

## Addendum C (2026-10-08): fix of the silent page cut before extraction, and one more DLinear run

Written and committed **before** the run described here. Coordinator task (Nachtrag C). Goal: a normal live run of Zeng DLinear finds the `drop_last` objection by itself and runs it as `protocol_change`. This is a function test of the feature, not a benchmark. No accuracy claim.

### Cause, from stored data only (no new Tavily calls)

Source: `events.jsonl` and `dossier.json` (`community.pages`, `hits`, `candidates`) of the Addendum A run (`20261008-151520`) and the clean Addendum B repeat (`20261008-164946`); FITS README read once from GitHub for free (not Tavily).

1. **Addendum A diagnosis was incomplete.** `community.what_others_found` passes `docs[:MAX_CHUNKS]` (12) to the extractor, in the order of the search results. In the run of Addendum A, 22 pages were read; the FITS README stood at index 12 (0-based), behind the cut. It never reached the extractor. This is why no FITS candidate was logged, and why the addendum could not explain it. The `names_paper` explanation (offline replay) is true but was not the live blocker. Nine more read pages (TSF-Skeleton, LTSF-Benchmark.md, exp_main.py, the LTSF-Linear repo page, LICENSE, README.md, issues #116, #118, #107) were cut the same way, 10 of 22 in total. Nothing in the log showed the cut.
2. **Addendum B rule works as designed but could not act.** In the clean B repeat, 17 pages were read and the last 5 (LTSF-Linear repo page, LICENSE, issues #107, #108, #117) were cut by the same slice. More important: **no hit among the 30 stored hits is the FITS README, SparseTSF or any page with `drop_last` text.** The only candidate (`decisionintelligence/TFB`) was not read, because the page filter wants method plus dataset in the text. The same four critique queries returned the FITS README in A and did not in B: Tavily search results vary between runs. The B rule (page names the method) was therefore never tested on live data.
3. `dossier.droplast_notes` (FITS README, SparseTSF, TFB) is a **fixed source list** in `droplast_note.py`, not a Tavily result. It must not be shown as a find. It stays as is.

So two obstacles: (a) a silent cut by arrival order, which can remove exactly the short, focused pages (a README bug notice) behind long papers; (b) search variance, which the code cannot remove but can make less costly.

### Fix (generic, no paper- or term-specific rule, no changed query text)

1. `community.rank_docs`: before the cut to `MAX_CHUNKS` (12 pages, unchanged, so extraction cost stays bounded), the pages read are ordered by a rarity score: for each setup/protocol keyword of `SETUP_KEYWORDS` that occurs in the page, weight `log((N+1)/(df+1))`, with `N` = pages read and `df` = pages containing the keyword; sum over the page's keywords; ties keep the arrival order. A keyword that stands on every page (hyperparameter, test set, seeds) weighs about 0. A page with a keyword that few pages share (here: `drop_last`, `leakage`) goes first. The keyword set is the existing `SETUP_KEYWORDS`; nothing new is hard-coded. With 12 pages or fewer the order has no effect on what is extracted.
2. `CRITIQUE_RESULTS = 8`: the four critique searches ask for 8 results instead of 5 (same credits: a basic search costs 1 credit, `logs/tavily_usage.jsonl` shows `credits: 1, depth: basic`). More recall against the search variance of point 2 above. Whether Tavily returns more than 5 is not checked offline.
3. Pages read but cut are now logged: `community.not_extracted` (dossier and `community` event). Nothing else in the event changes (`web/` is not touched).

Not changed: `names_paper` and the page-level rule of Addendum B, the quote check, the page filter, the noise filter, objection classification, the fixed demotion rules, the `protocol_change` rule (`effect >= tolerance/4`), the experiment limits, the harness. No query text.

Known limits, stated now: (i) ranking by rarity is a heuristic; a long paper with a rare keyword can still outrank a README and vice versa. (ii) With IDF over a small set (N around 15 to 25) the weights are coarse. (iii) The Addendum B page-level rule still lets generic findings through on pages that name the method (3 such findings in the clean B repeat); this fix does not tighten it, and precision of the community round is **not improved** by it. (iv) The ranking may cut pages that gave kept findings before; that is reported if it happens.

### Offline tests (no network)

`tests/test_community.py`, new: `rank_docs` puts the rare-keyword page first and keeps ties in order; a scenario modelled on Addendum A (12 generic pages naming DLinear and ETTh1 with common keywords, then a page with a verbatim excerpt of the FITS README at index 13 of the critique results) yields the finding with `pass_reason = page_names_method`, the page is among the 13 read; with the old order (`rank_docs` replaced by identity) the same input yields no finding and `not_extracted == [FITS url]`; a page that names neither the method nor the dataset (a KAN-paper stand-in with `drop_last` in its text) is not read; 12 generic pages alone give no finding and no candidate; the critique search is asked for `CRITIQUE_RESULTS >= 8` results; the FITS-style quote passes the fixed `protocol_change` validation, a generic "baselines are quoted from their publications" quote is demoted to `not_measurable`. The text of the FITS excerpt is verbatim from the public README; the filler pages are synthetic.

### Run (phase D of this addendum)

One live run: `LLM_PROVIDER=tokenfactory TAVILY_MODE=live SANDBOX_BACKEND=nebius nohup uv run python scripts/run_objection_live.py --paper zeng_dlinear_2023 --claim dlinear_etth1_96_mse --fresh-community`, no `timeout` wrapper, full mode (5 seeds), product configuration with `objection_experiments=True`, no LLM cache. Sandbox backend `nebius` (a parallel Docker run on the laptop forbids `local`; if Nebius fails, ask the coordinator, do not switch).

- **Caps for Addendum C:** Token Factory **0.25 USD gross**, Tavily **40 credits** (counted from the start timestamp of the run in `logs/tavily_usage.jsonl`; other sessions share that log, so any foreign call in the window is reported). Observed full runs cost 0.18 (A) to 0.33 USD gross (B); the 0.25 cap is therefore a **target, not a hard stop**: an abort in the middle of the run would give no result, so the run keeps the standard brake `MAX_COST_PER_RUN_USD=0.50` and an excess is reported as it is. Expected Tavily use about 18 to 24 credits (4 critique searches cost the same with 8 results).
- **Number of runs: 1.** A repeat only after an infrastructure failure (sandbox setup, API outage, the known role-2 `ExtractionInvalid` crash before the community round) and only if the caps allow. Not a repeat to improve a result.

### Acceptance criteria (all required; otherwise "not met")

1. The run completes with a verdict and writes `dossier.json`, `events.jsonl`, `zeugnis.md`.
2. The community round, not `droplast_notes`, delivers a finding whose quote is verbatim on a stored page and is about the test loader dropping the last batch (a `kind = protocol note` finding with its URL).
3. The objection round classifies a finding as `protocol_change` (after the fixed demotion) **without any manual input**, and `objection_checks` holds that entry with type `protocol_change`.
4. Both numbers are measured: the entry shows the full-test-set value and the `drop_last` value from at least 2 valid main seeds, and its status follows the fixed rule applied to the stored numbers (recomputed in the evaluation).
5. The main verdict (`urteil`, mean, std) rests on main attempts only.

Expectation from the manual run, not a criterion: full 0.3793, truncated 0.3754 (batch size 128), paper 0.375; I expect `widerlegt` or `bestaetigt` depending on the batch size named in the quote (effect threshold 0.0025; 0.3793 vs 0.3754 is 0.0039 at batch size 128).

### Honest expectation

Whether the FITS README (or another page with the statement) is returned again is up to Tavily: one of two live runs returned it. The fix removes the silent cut and widens the search a little; it cannot force the hit. If no such finding appears, the result is "not met", reported with hits, `not_extracted` and drop reasons from the log. No second run for that.

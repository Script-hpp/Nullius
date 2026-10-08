# Tavily experiments, part 1 result (DLinear fixture rehearsal)

Pre-registration: `benchmark/experiments/tavily_experiments_test.md` (part 1). Run: `runs/zeng_dlinear_2023/20261007-224416` (started by the previous worker, finished 2026-10-07 23:51; completed, so it was not repeated). Fixture mode, Token Factory, Nebius sandbox, full mode, one run. Not a benchmark, no accuracy claim.

## Criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Run completes; `dossier.json` with `objection_checks`, `events.jsonl` with the four new event types, `zeugnis.md` with "Objections checked" | **not met** (strict) | `run_end` present; `objection_checks` 4 entries; `zeugnis.md` line 19 has the section. Events: `objection_classified` x4, `objection_checked` x4, `objection_round_end` x1; the fourth type `objection_experiment` is emitted only when an experiment runs, and 0 ran (criterion 3), so only 3 of 4 types occur. |
| 2 | Every entry keeps a verbatim quote found in the stored fixture page and its URL | **met** | all 4 quotes and URLs found in `benchmark/fixtures/role3_zeng_objections.json` (substring check) |
| 3 | At least 1 entry non-`not_measurable` and a `seed_variance`/`parameter_change` status resting on at least 2 valid own-branch measurements | **not met** | types: 1 `reported_number`, 3 `not_measurable`; 0 `seed_variance`, 0 `parameter_change`; `objection_round_end`: `n_experiments = 0`. The only measurable finding (issue #117) is a reported number, no run by rule. |
| 4 | Each status follows the fixed rule applied to the stored numbers | **met** | obj1: `same_setup = "no"` -> `offen` (rule needs `yes`); also `|ours - reported| = 0.033 > tol 0.01`, `|ours - target| = 0.084 > tol`, so no rule branch for `bestaetigt`/`widerlegt` applies. obj2-obj4: `not_measurable` -> `nicht pruefbar` |
| 5 | Main verdict from main attempts only | **met** | verdict `nicht reproduziert`, mean 0.4592, std 0.0353; recomputed from the 5 main `abweichung` attempts (seeds 0-4: 0.4665, 0.5173, 0.4314, 0.4457, 0.4350) = 0.4592. `objection_checks` entries have empty `attempts`. |

Part 1 overall: criteria 1 (event type `objection_experiment` absent because no experiment ran) and 3 not met, so part 1 is **not met** as pre-registered. Criteria 2, 4, 5 met.

## Numbers

- Cost of run: 0.2018 USD net, 0.2401 USD gross (`run_end`). Objection round: 0.0019 USD net, 8.9 s, 4 findings, 0 experiments. Tavily credits: 0 (stored fixture).
- `logs/costs.jsonl` does not exist in this worktree (`logs/` is an empty directory here, not a symlink), so costs come from events only. The cap bookkeeping for part 2 must read the main checkout's log or sum events.
- Counts per type: reported_number 1, not_measurable 3. Per status: offen 1, nicht pruefbar 3.

## Observations

- Expectation in the pre-registration was a `parameter_change` for "input window 336 vs 96". The stored findings contain no such objection (obj4 mentions "bigger window" without a value, correctly `not_measurable`). Stored fixture offers no measurable seed/parameter finding for DLinear, consistent with `results/role3_dlinear_stage1.md`.
- Issue #117: model classified `same_setup = "no"`, not `unknown` as expected. Not an error under the pre-registration (only `yes` would be). Status `offen` either way.
- obj3 `type_reason` is empty and note "classification missing or invalid": classifier output for that finding was invalid and fell back to `not_measurable`.
- The main run itself did not reproduce the target (0.459 vs 0.375, 9 of 13 attempts were `fehler` with repair, 4 seed attempts `abweichung`). Within scope of this test; not a feature problem. It did not re-run.
- Conclusion: pipeline runs end to end and demotes correctly, but part 1 cannot show a measured objection experiment on DLinear. Seed/parameter path remains untested on real data until part 2.

## Cost bookkeeping note

Part 1 cost 0.2401 USD gross (0.2018 net) per the `run_end` event. It is missing from `logs/costs.jsonl` because the worktree `logs/` was an empty directory, not a symlink, during that run. For the part 2 cap (1.00 USD gross total), count 0.2401 USD as already spent. Remaining at the start of part 2: 0.7599 USD gross. Tavily credits used in part 1: 0.

# Part 2 (live runs, approved by coordinator 2026-10-08; interim, written by the resume worker)

Runner: `scripts/run_objection_live.py` (product config: `graph_repair=True`, `objection_experiments=True`, `TAVILY_MODE=live`, Token Factory, Nebius sandbox, full mode, no LLM cache). Worktree `logs/` is now a symlink to the main checkout's `logs/`.

| Run | Dir (`runs/...`) | Verdict | Gross cost | Tavily credits (own, from `tavily_usage.jsonl`) | Objection checks |
|---|---|---|---|---|---|
| Kipf run 1 | `kipf_gcn_2017/20261008-122632` | reproduziert 0.8151 +/- 0.0107 | 0.0809 | 7 | 0 (community round served from `logs/community_cache.json`, `mode=cache`, entry from 2026-10-02: 0 verified findings, 2 dropped) |
| Kipf run 2 (`--fresh-community`) | `kipf_gcn_2017/20261008-130547` | teilweise 0.8000 +/- 0.0138 | 0.0648 | 13 | 0 (community `mode=live`, 3 searches, 0 verified findings, 1 dropped) |
| DLinear | `zeng_dlinear_2023/20261008-133630` | reproduziert 0.3774 +/- 0.0003 | 0.2226 | about 14 | 0 (community check failed: `ConnectionError ... RemoteDisconnected` from Tavily, run continued; `dossier.community = None`) |

Deviation (reported): the pre-registration said role 3 "needs a live search (no fixture exists)". A 7-day community cache entry for 1609.02907 existed in the global `logs/` and served run 1. With coordinator approval run 2 bypassed the cache (script-level flag, global cache file untouched). This is a configuration fix, not a rerun to improve a result; both runs are reported. Kipf verdict differs between the two runs (reproduziert vs teilweise): independent generations, no feature involvement.

Totals after the three runs: Token Factory 0.2401 (part 1) + 0.0809 + 0.0648 + 0.2226 = 0.6084 USD gross of 1.00; Tavily about 34 credits of 60 (counts are from the shared global log, other sessions' calls excluded by hand; estimator rounds up).

## Part 2 criteria

1. Run completes with verdict, `objection_checks` not empty or log shows why: Kipf run 1 and 2: verdicts present; empty with reason in the log (0 verified findings), **met** by the "or log shows why" clause. DLinear: verdict present, empty because the community search crashed on a Tavily connection error, the log shows a warning but no finding was evaluated; **not met in substance** (infrastructure failure).
2. Criteria 2, 4, 5 of part 1: no entries exist, so vacuous; not evidence for the feature.
3. `objection_round_end` seconds and cost: no event was written in any run (the round returned early without findings). No data.

Result so far: the live path never produced a checked objection, so the acceptance condition (exported run with an "Objections checked" block) is **not reached**. Nothing here shows that the feature works or fails on live findings; it shows that for GCN/Cora the live community search yields no verified, measurable finding, and a Tavily connection error cost the DLinear round.

## Offline diagnosis (no Tavily, no Token Factory; written after coordinator chose "stop paid runs")

Limit of the evidence: the dossier stores per community page only `url`, `n_chars`, `sha256`, not the Tavily snippets or the LLM's dropped candidates. What follows uses those fields plus free web fetches of the page URLs (WebFetch, not Tavily).

**Kipf run 2 (live, `mode=live`).** 3 searches (title, method+dataset, repo issues of `tkipf/gcn`). `select_docs` kept 2 pages:
- `arxiv.org/html/2102.12557v1` (22,084 chars): "Benchmarking Graph Neural Networks on Link Prediction" (Wang, Vinel). Link prediction across datasets; abstract names no Cora, seeds or splits. Not a node-classification protocol note for this claim.
- `github.com/tkipf/gcn/issues/14` (143 chars): "Can't run on Nell dataset #14", a `load_data` crash (row index out of bounds). Not a measurable objection; Tavily returned only 143 chars of it (title level), so the extractor saw almost nothing.
The LLM extractor produced 1 candidate that was dropped. The code gives two possible reasons (`extract_objections` quote check, or `names_paper` for a statement naming neither method nor title); the reason is not logged, so which one fired cannot be shown.

**Kipf run 1 (cache entry from 2026-10-02).** 7 pages kept, 2 candidates dropped, 0 findings. Pages (titles by free fetch): 2009.13734 (GCN plus side information, new method), 2207.05669 (GCN to large-scale GCN, reproduces Kipf's numbers, abstract names no seed/split variance), 1803.04489 (Probabilistic and Regularized GCNs, reproduces Kipf, alternatives do not help), plus a Reddit paper-notes thread, an OpenReview forum page, `eric-yyjau/paperNotes/issues/2`, `tkipf/gcn/issues/14`. All are re-implementations or follow-up methods, not critiques of the protocol.

**Would a human see a measurable objection?** In these sources: no for seed dependence, split or hyperparameter. Two papers reproduce Kipf's numbers (supporting, not objecting); I did not read their full text, so a seed remark inside them is possible but not shown. The Kipf GCN claim has no well-known public objection of the seed/split kind in what Tavily returned. The known critique (Shchur et al. 2018, 1811.05868) is a separate catalog paper, not returned for the Kipf title query.

**DLinear connection error.** `CountedClient._call` (`src/repro_agent/tavily_usage.py`) had no retry. The Tavily SDK uses `requests`; `RemoteDisconnected` surfaces as `requests.exceptions.ConnectionError`, the community check caught it and gave up (it is a best-effort add-on), so the whole objection round was lost. Fixed: up to 3 attempts with 2 s / 4 s backoff for `ConnectionError`, `Timeout`, `ChunkedEncodingError` only; other errors (key, limit, 4xx) are not retried; each attempt is logged as one call (conservative credit count). Tests: `tests/test_tavily_usage.py` (retry succeeds on third attempt with 3 log lines; gives up after three; non-transient error not retried). The LLM client's own 5-attempt retry is separate.

**Recommendation.**
1. For GCN/Cora, "sources contain nothing measurable" is the best reading of the stored evidence; no filter or query change is justified by it. Do not loosen `names_paper`/quote checks to get findings.
2. Keep the Kipf query as is, but if a measurable finding is wanted for the acceptance condition, the paper choice matters more than the filter: DLinear's stored fixture has a real issue (#117, reported number 0.426), and Shchur-type papers have a known critique. One live DLinear run with the retry fix is the cheapest next step (about 0.22 USD gross, about 14 credits; 0.39 USD / about 26 credits left under the caps). Not run, awaiting decision.
3. For diagnosability, log the Tavily result URLs and titles and the reason each LLM candidate was dropped (small change in `community.what_others_found`). Without it the "1 dropped" cannot be explained after the fact.

# Tavily experiments, addendum B result (page-level names_paper for setup findings)

Pre-registration: `benchmark/experiments/tavily_experiments_test.md`, "Addendum B", committed in `e04f081` before the run. Post hoc rule change, approved by the user on 2026-10-08.

## Outcome in one line

The live run **crashed before the community round** (infrastructure failure), so there are **no findings, no kept/dropped data, no `protocol_change` entry and no measured numbers**. The rule change is untested on live data. Acceptance is not reached.

## What happened

- Command: `LLM_PROVIDER=tokenfactory TAVILY_MODE=live uv run python scripts/run_objection_live.py --paper zeng_dlinear_2023 --claim dlinear_etth1_96_mse --fresh-community` (run dir `runs/zeng_dlinear_2023/20261008-154446`, no dossier).
- Error in role 2 (target extraction, `claims_source.extract_target`): `ExtractionInvalid: Modellantwort enthaelt kein JSON` after the built-in retries. This is the model answer, unrelated to the new code.
- Cost of the failed run: 0.0221 USD net, 0.0263 USD gross, 2 Tavily credits (window from start timestamp 1791467086). Totals against the caps: about 0.95 of 1.25 USD gross, about 61 of 80 credits.
- The addendum allows one repeat after an infrastructure failure only with coordinator approval. I asked via `orca orchestration ask` (msg_18d9e7fcd566); no answer arrived within about 35 minutes, so I did **not** repeat the run.

## Not shown

Found / kept / dropped counts, whether the FITS `drop_last` quote passes with `pass_reason = page_names_method`, and the measured full vs truncated test-set numbers: none exist. Whether a repeat would reach the FITS README is open.

## Offline evidence

Unit tests (`tests/test_community.py`, four new): setup finding passes by page text (`pass_reason = page_names_method`); dropped when the page does not name the method (select_docs forced to isolate the rule); non-setup finding keeps the statement rule; quote not on page stays `dropped_quote_not_in_page`; raw candidates with outcome stored. Community, objection and drop_last test files: 93 passed. Full suite (minus `test_cora_baselines_and_faithful_gcn`, failing before this work): not finished within the tool time limit in my last attempt; first 407 tests passed.

## Honest note on the rule

Because pages are read only if they already name title, arXiv ID or (method hits) method plus dataset, the page-level test is nearly always true for a read page. The rule effectively lets any setup-topic finding on a page about the method through. Stated in the pre-registration, point 4.

## Repeat attempt (2026-10-08, coordinator task)

- Crash analysis of the first run: role 2 uses model role `fast` (Lightning). Three attempts (temperatures 0.0/0.1/0.2, `max_tokens` 6000, then doubled on `finish_reason: length`) gave 6000 (length), 10607 (stop) and 12000 (length) completion tokens, none with a JSON object containing `quote`. Lightning writes its reasoning into the answer; this is the known quirk, not a new bug. `claims_source.extract_target` already retries with other temperature and larger limit, so no code change was made. Open robustness idea (not built): escalate the last attempt to the `code` role (Super).
- Repeat run (same command, run dir `runs/zeng_dlinear_2023/20261008-162236`): role 2 passed this time (target 0.375 verified), role 4 gave 110 kept / 11 dropped. The run was then **killed by my own `timeout 590` wrapper** during the seed-0 repair loop (attempt 3). This is an operator error, not an infrastructure failure. The community round never started: **no found/kept/dropped data, no `protocol_change` entry, no full vs truncated numbers.**
- Spent in this addendum since its start: about 0.231 USD gross, 10 Tavily credits (about 1.15 of 1.25 USD and about 69 of 80 credits in total). Another full run (about 0.18 USD gross) would pass the 1.25 USD cap, so I asked the coordinator to raise it (msg_bca0bd43665a, no answer after 15 minutes) and did **not** run again.
- Status: acceptance not reached; Addendum B rule still untested on live data.

## Clean live repeat (2026-10-08, coordinator task, no timeout wrapper)

Run dir `runs/zeng_dlinear_2023/20261008-164946`; same command as before, started with `nohup`, ran to the end (`run_end`). Status: **completed**. Main verdict: `reproduziert`, mean 0.3793, std 0.0021 over 5 of 5 seeds (target 0.375 +/- 0.01), method fidelity not checked.

### Community round (raw candidates stored, `community` event)

- Found: 15 raw candidates on 4 pages. Kept: 9 (`n_findings` 9). Dropped: 6.
- Kept by the new rule (`kept_page_names_method`): 3, all on arXiv papers that name DLinear (2606.27282v2, 2605.02689v1, 2606.27282v1), all generic "baselines are quoted from original publications / literature baselines use other hyperparameters" statements. Kept by the old rule (`kept_statement_names_method`): 5 GitHub issues of cure-lab/LTSF-Linear plus 1 arXiv 2604.23968v1 statement (no multi-seed baselines).
- Dropped: 5 `dropped_names_neither_method_title_nor_arxiv_id` (arXiv 2604.23968v1, a page that does not name DLinear; its statements are about DecompKAN) and 1 `dropped_quote_not_in_page`. Other 4 drops were duplicates/page-level entries inside `dropped_items`.
- Honest note: the page-level rule let 3 generic findings through that the old rule would have dropped. They are about the cited paper's own baseline reporting, not a DLinear-specific protocol claim, as predicted in pre-registration point 4.
- **The FITS README was not returned this time; no `drop_last` finding exists.** The Addendum B expectation (FITS finding passes) is not tested.

### Objection round

9 objections classified: 3 `widerlegt` (obj2, obj3 `seed_variance`, one experiment; obj6 `reported_number`), 6 `nicht pruefbar` (`not_measurable`, two of them with "classification missing or invalid"). 1 own experiment. Round took 65 s, 0.0047 USD net.

### protocol_change

**Did not run.** No `protocol_change` objection was produced, no event contains it. Therefore no protocol_change numbers exist. Separate, pre-existing dossier field `droplast_notes` (built from FITS README / SparseTSF sources via the older setup-details path, not a community finding of this round) holds: full test set MSE 0.3793, `drop_last` truncated test set 0.3754 (97 of 2785 windows dropped, batch size 128), claimed 0.375, both within tolerance 0.01. Do not read this as a `protocol_change` result.

### Cost

Run cost 0.2732 USD net, 0.3251 USD gross; Tavily 18 credits (from `logs/tavily_usage.jsonl`, since start timestamp 1791470986). Totals for the addendum: about 1.47 USD gross (cap 1.45, exceeded by about 0.02 because the previous estimate of 1.15 was approximate) and about 87 of 95 credits.

### Status

Addendum B ran live once without infrastructure failure. The rule change works mechanically (3 page-level keeps, raw candidates stored), but it did not produce the targeted FITS `drop_last` finding and `protocol_change` never ran. Acceptance for the `protocol_change` path is not reached. No further runs.

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

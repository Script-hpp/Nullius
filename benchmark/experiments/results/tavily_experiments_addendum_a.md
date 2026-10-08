# Tavily experiments, addendum A result (generic query pivot, DLinear live run 5)

Pre-registration: `benchmark/experiments/tavily_experiments_test.md`, "Addendum A", committed in `321228e` before the run. Run: `runs/zeng_dlinear_2023/20261008-151520`, `dlinear_etth1_96_mse`, 5 seeds, full mode, `--fresh-community`, Token Factory, `TAVILY_MODE=live`, Nebius sandbox. Not a benchmark, no accuracy claim. Post hoc pivot after coordinator web research (stated in the pre-registration).

## Outcome in one line

The community round ran live with the new queries and read 22 pages, but produced **0 findings**, so **no `protocol_change` entry and no objection experiment**. The `drop_last` finding did not appear. Acceptance (exported run with a checked objection) is **not reached**.

## Numbers

- Main verdict: `teilweise`, 0.3901 +/- 0.0006 (target 0.375, tolerance 0.01), 4 of 5 seeds measured in the final attempt set after repair of seed 0 (`fehler` then `abweichung`). Not related to the feature.
- Cost: run 0.1538 USD net, 0.1830 USD gross (`run_end`). Objection round: no `objection_*` event was written (round returns early without findings), so no seconds/cost for it. Total Token Factory: 0.733 + 0.183 = 0.916 USD gross, plus about 0.002 USD net for the diagnostic below, about 0.92 of 1.00. The 0.90 stop rule applies to starting a run; it was 0.733 at start.
- Tavily credits in this run (from `logs/tavily_usage.jsonl`, window of the run): 18 (2 extracts for roles 2/4, 11 searches, 1 crawl; the community round used 7 searches + 1 crawl = about 8, the rest roles 1/2/4/5). Total about 41 + 18 = 59 of 80 (stop rule 72 not reached).
- Community event: `mode=live`, 5 query groups (title search, method search, 4 critique templates, repo search, crawl of `cure-lab/LTSF-Linear/issues` limit 3), 35 hits, 22 pages read, 12 not read (collections, pages above 60,000 characters, or pages naming neither paper title/ID nor method plus dataset), 2 candidates dropped, 0 findings.
- New queries as sent: `DLinear ETTh1 reproduce results differ`, `DLinear bug evaluation`, `DLinear baseline re-evaluation fixed`, `we re-run DLinear on ETTh1 and compare with the reported numbers`.

## What Tavily returned that matters

- The new queries did surface follow-up material: `VEWOXIC/FITS` README (11,204 chars), `arXiv 2501.01087` (html and pdf), `2605.24381`, `2605.02689`, `2312.06786`, `wayne155/FAN`, `cure-lab/TSF-Skeleton`, several DLinear re-implementations, and three **numbered issue pages with text** of `cure-lab/LTSF-Linear` from the crawl (#116, #118, #107, 514 to 1,713 chars each; previously only the list headings came back). The crawl ran because the search returned only the list page; no short numbered issue hit needed the Extract body fetch.
- No discussion-forum page (`discuss.dgl.ai` etc.) was returned for DLinear (expected: it is a time-series paper).

## What was dropped, and why

- 2 candidates, both `names_neither_method_title_nor_arxiv_id` (`names_paper`, unchanged): (1) `arxiv.org/html/2605.02689v1`: literature baselines "may not be directly comparable due to possible differences in hyperparameters" (generic); (2) `github.com/wayne155/FAN`: "scaling using the entire dataset instead of only the training set creates a label leak problem" (a statement about that repo's code, not about DLinear). Both drops are correct under the rule. 0 drops by quote check.
- The three crawled issue pages gave no finding (extractor returned nothing measurable; headings: memory leak in `test()`, a shape error, an environment question).

## Diagnosis of the missing drop_last finding (free web fetch plus about 0.002 USD, outside the pre-registered run)

The FITS README was in the 22 pages read. The extractor candidates of the live run for that page are not stored (only drops are), so I replayed the extractor once on the README text fetched from GitHub (not Tavily; Tavily's copy is 1,424 chars shorter and may differ):

- The extractor **does** return the drop_last statement with a verbatim quote ("the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data ...").
- That statement and quote name neither "DLinear", "LTSF-Linear" as method name, the paper title, nor the arXiv ID (README of a different method, FITS), so the unchanged `names_paper` check drops it: `names_paper(...) = False` for all 5 candidates of that page.
- In the live run no FITS candidate appears among the 2 logged drops, so either the live extractor output differed (temperature 0 is not deterministic across calls) or Tavily's page text differed. This cannot be resolved without storing the raw candidates; not claimed either way.

Reading: even when the page is found and the quote is verified, a generic cross-paper finding (a follow-up paper saying "the LTSF-style test loader drops the last batch") is excluded by the rule "finding must name the method, title or arXiv ID". The hoped-for finding is therefore blocked by a check that the addendum keeps unchanged by design, not only by the search. Loosening it (for example: accept a finding that names the paper repo's family, or an organisation-level source naming `LTSF` or `DLinear` anywhere on the page) would be a rule change after seeing results; it needs the coordinator's decision and a new pre-registration.

## Feature code (tested offline)

- `protocol_change` classification, fixed demotions, rule and Evidence line have unit tests (`tests/test_objection_experiments.py`, six new tests: validation against quote, task and metric; no branch, no code, main attempts untouched; rule boundaries at tolerance/4; offen without stored metrics; experiment limit unaffected; ETTh1 declares `drop_last`). Community changes have five new tests (`tests/test_community.py`: generic templates and forum domains, critique hits need method plus dataset in the page, short issue hits get bodies via a limited Extract, list page alone triggers the crawl, extract failure keeps paid hits). Full suite: 928 passed, 2 skipped, 5 xfailed (1 deselected: `test_cora_baselines_and_faithful_gcn`, failing before this change, untouched code).
- **Not shown on live data:** `protocol_change` has never run on a real finding. The tests use stored-metric fakes.

## Honest summary against the pre-registered expectation

The expectation said the finding was hoped for and might not appear. It did not appear in the final findings. Search was not the only obstacle: the page was probably read, and the quote was extractable, but the unchanged `names_paper` rule drops it. Nothing here shows or refutes that Tavily-found `drop_last` evidence can drive a harness-side re-evaluation on a live run. No result was re-run, and no check was loosened.

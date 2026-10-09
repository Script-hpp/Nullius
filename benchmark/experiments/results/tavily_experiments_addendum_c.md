# Tavily experiments, addendum C result (page ranking before the extractor cut, DLinear live run 8)

Pre-registration: `benchmark/experiments/tavily_experiments_test.md`, "Addendum C", committed in `0995bc0` before the run. Run: `runs/zeng_dlinear_2023/20261008-210809`, claim `dlinear_etth1_96_mse`, 5 seeds, full mode, `--fresh-community`, Token Factory, `TAVILY_MODE=live`, Nebius sandbox, started with `nohup`, no timeout wrapper. Function test, not a benchmark, no accuracy claim.

## Outcome in one line

The run completed, but the community round returned **no page that states the `drop_last` problem**, so **no `protocol_change` entry exists and nothing was measured for it**. Acceptance is **not met** (criteria 1 and 5 met; 2, 3, 4 not met). The fix did what it was built for (pages are no longer cut silently by arrival order), but it could not create a hit that Tavily did not return.

## Criteria

1. Run completes with verdict, `dossier.json`, `events.jsonl`, `zeugnis.md`: **met**. Verdict `nicht reproduziert`, mean 0.4296, std 0.0099 (target 0.375, tolerance 0.01), 5 seeds; seed 0 needed many attempts (7 `fehler` attempts in total, repair loop took about 40 of the 60 minutes). Unrelated to the feature.
2. Community finding about the test loader dropping the last batch, verbatim on a stored page: **not met**. 11 findings, none about `drop_last` or the last test batch.
3. Objection classified as `protocol_change` without manual input: **not met**. 11 objections: 10 `not_measurable` (`nicht pruefbar`), 1 `reported_number` (`offen`, "the quote does not state the same setup as our task": the LTSF-Linear issue with mse 0.426 vs 0.386). 0 experiments, round took 30.8 s, 0.0059 USD net.
4. Both numbers measured: **not met** (no entry).
5. Main verdict rests on main attempts only: **met** (no objection experiment ran).

## What the search returned

- 45 hits (the 4 critique searches ask for 8 results now; before, 30 hits). **No hit is the FITS README (`VEWOXIC/FITS`), SparseTSF or any other URL I know to state the `drop_last` bug.** `decisionintelligence/TFB` came back again and was again not read (page filter: it does not name method plus dataset). So over three live runs with the same query set, the FITS README came back once (Addendum A), not in B, not in C: search variance, as predicted. Page texts of hits are not stored, so I cannot rule out `drop_last` text on one of the 45 pages from the log alone; the stored extractor candidates (below) contain none.
- 20 pages read, 12 sent to the extractor, 8 cut and now logged in `community.not_extracted`: `debi2023-group3/Transformer-based-solutions...`, `LTSF-Benchmark.md`, `awslabs/gluonts/issues/3046`, `exp/exp_main.py`, `LICENSE`, LTSF-Linear issues #107, #108, #117. The ranking moved these behind arXiv papers that contain rarer setup keywords (for example optimizer or warmup passages). These cut pages are the same kind of page (short repo files and issues) that the old order cut in A and B, so the ranking did not rescue them in this run. Whether the old order would have been better here is not tested.
- Candidates (15 raw): 11 kept (6 by page-level rule, 5 by statement-level rule), 4 dropped for naming neither method, title nor arXiv ID. The page-level rule again let generic statements through (baselines quoted from publications, warmup budget and optimizer rate in an online-adaptation paper on arXiv 2609.01126, a future-informed label in 2609.39789). This is the precision loss that Addendum B point 4 and Addendum C limit (iii) predicted; the fix did not change it, and the extra search results added new such pages. **Precision of the community round got worse in the sense of more off-topic kept findings (11 vs 9), not better.**

## Cost

- Run: 0.2012 USD net, **0.2395 USD gross** (`run_end`). Under the 0.25 target and under the coordinator's hard cap of 0.40 USD gross. The coordinator's message with the 0.40 cap (`MAX_COST_PER_RUN_USD=0.40`) arrived after the run had started; the run used the `.env` brake of 0.50, which it did not approach.
- Tavily: **19 credits** in the window from the run start timestamp 1791486488 (`logs/tavily_usage.jsonl`: 2 extracts x 2, 11 searches, 1 crawl x 3, 1 more search; 15 calls, no foreign call visible in the window). Under the 40-credit cap.
- Model calls in the community round and the objection round are small (objection round 0.0059 USD net).

## Offline checks

Full test suite before the run: 934 passed, 2 skipped, 5 xfailed; failures: `tests/test_harness_baselines.py::test_cora_baselines_and_faithful_gcn` (failing before this work, deselected) and `tests/test_nebius_sandbox.py::test_get_file_large[10]` and `[50]` (live Nebius download of 10 and 50 MB files, `ContreeTransportError` / "peer closed connection without sending complete message body"; a sandbox transport problem, not touched by this work; the live run did not hit it). `tests/test_community.py`: 40 passed incl. six new tests of Addendum C.

## Honest reading

- The silent cut was a real defect in the code, found in the stored data of Addendum A (FITS README at index 12 of 22 read pages, behind `docs[:12]`), and is fixed and visible now. It is **not** the reason for this run's miss: this time the pages that state the problem were not returned at all.
- The hoped-for video scene (a Tavily find becomes a `protocol_change` experiment automatically) is **not shown live**: in 3 of 3 live runs with the new query set, `protocol_change` never ran. It runs offline on stored predictions and in unit tests, and it works by hand (full 0.3793, `drop_last` 0.3754, paper 0.375, `droplast_notes` of the Addendum B run). Presenting the scene as automatic live behavior would overstate the evidence.
- What could still raise the hit rate, not built and not tested (each needs a new pre-registration and a Tavily-credit budget): follow `github.com/<other repo>` links or a citing-paper search from the paper's own repo issues; a read-priority rule that also reads pages naming `LTSF` (the repo family) without dataset; reading hits from the TFB README. All are guesses about Tavily's results; none was run.

No second run for this addendum (pre-registered: one run; no infrastructure failure occurred).

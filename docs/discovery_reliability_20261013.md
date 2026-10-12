# Automatic Discovery reliability (2026-10-13)

User requested automatic Discovery reliability, rather than a separate paper-URL
flow. Research and fixes follow the deployed zero-candidate search documented in
`discovery_horizon_20261012.md`.

## Primary-source findings

| Paper | Source evidence | Decision |
| --- | --- | --- |
| [SARAF, 2606.04135](https://arxiv.org/html/2606.04135) | Table 4: SARAF's own ETTh1 MSE is **0.371**, horizon 96, look-back 720; RAFT is 0.366. Author code: https://github.com/ShiqiaoZhou/SARAF | Compatible quoted claim exists. No reproduced result is promised. |
| [GLinear, 2501.01087](https://arxiv.org/html/2501.01087) | Table II: GLinear's own ETTh1 MSE is **0.3820**, horizon 96, input 336. Author code: https://github.com/t-rizvi/GLinear | A second compatible, simpler claim exists. The old saved experiment stays separate. |
| [Time-VLM, 2502.04395](https://arxiv.org/html/2502.04395) | Abstract/method explicitly depend on pretrained vision-language models | Outside the current from-scratch CPU scope; do not spend an extraction slot on this known dependency. |
| [Data Embeddings, 2505.20716](https://arxiv.org/html/2505.20716) | Times2D is a cited earlier model in a comparison/study, rather than this paper's newly proposed method | Do not misrepresent its baseline number as a new proposed-method claim. |

SARAF's actual abstract does **not** name ETTh1. The old abstract-only prefilter
could therefore discard it when a Tavily snippet omitted the dataset, despite
the complete source containing a compatible result.

## Changes

- Keep the normal dataset-scoped Tavily search. Read its bounded results against
  full source when the abstract omits the benchmark. Confirm the benchmark in
  readable full text before asking the model to assess it.
- Deprioritize explicitly dependent pretrained-model abstracts and reject those
  clear dependencies before source/model extraction. Public-code ranking remains
  stable. Search/extraction/candidate limits are unchanged.
- `forecast_claim.py` selects a unique proposed-method MSE/horizon-96 claim from
  an aligned main LaTeX table with a supported, caption-evidenced input length.
  Proposal evidence is an `(ours)`/proposed header or a proposal statement in
  the paper abstract. Unknown layouts and conflicting source results return no
  deterministic claim; ablation, few-shot and univariate tables are excluded.
- The existing model still assesses setup and feasibility. Its empty, baseline,
  average or wrong-horizon numerical selection cannot replace an independently
  verified main-table claim. Explicit feasibility/protocol rejection, provider
  failures, billing/budget guards and truncated/unreadable output stay guarded.
  Empty output with a verified source claim leaves the protocol **unknown**;
  no protocol or method-fidelity certification is invented. That empty output
  does not need another numerical model call.
- Fresh source selection is logged as `source_claim`, including model method and
  target, and carries `extraction_basis` in the candidate. It passes the existing
  quote, number, column, protocol, repository and persistence checks.
- The table checker understands braced cell line breaks (`shortstack`), nested
  `multicolumn`, metric headers without the word Metric, and bounded multirow
  dataset labels. It never carries a label beyond its declared row span. Full
  verbatim dataset-group quotes retain original case/spacing.
- An explicitly cited earlier method without proposal evidence is rejected in
  forecasting discovery. The graph method-name check remains unchanged.
- Cache v6 re-examines old extraction rejections. Accepted historic candidates
  remain importable. Historic claims, targets, runs, verdicts and demo recordings
  are not rewritten, and known papers are not injected into live search results.

## Evidence and acceptance scope

Real public-source table excerpts and abstracts are test fixtures, **not** live
search fallbacks. `test_discovery_reliability.py` executes automatic Discovery
with the three incident papers, real parsing/validation, actual candidate
persistence and the production reproduction adapter. Only network/provider/repo
availability is mocked. Four model cases (empty, average, baseline, wrong
method/horizon) all yield the correct stored, runnable SARAF 0.371 claim. GLinear
also reaches a stored runnable 0.382 claim with unknown setup warnings retained.
The source selector independently verifies both original tables.

324 focused backend tests and 140 frontend tests passed; production build passed
and lint has no errors (existing warnings remain). Two previously known query-
limit tests remain deselected: they expect three queries while production uses
five, unchanged. Four production browser cases (desktop/mobile, mocked live/
saved candidates) show both correct targets, verified columns and enabled check
buttons, with no overflow or page errors. No paid search or experiment was run.

This proves finding and storing compatible claims from these source inputs,
not that every future web search will return one. Tavily must still return a
compatible paper, sources/provider must be available, and budget/setup/feasibility
guards may legitimately stop a run. Unknown table layouts remain guarded; the
regex proposal/dependency checks are limited, not universal scientific classifiers.
CPU completion and reaching a paper's target require a separate real experiment.

## Deployment

User deploys after the main push. No new environment variable, budget increase,
data-volume change or new user flow. After deployment, use the existing normal
ETTh1 live search when the daily gate permits. Inspect the **new** source-selected
claim; the previously saved SARAF 0.366 entry remains historical and should not
be confused with a corrected fresh claim.

## Live confirmation and protocol quote follow-up

The user's next deployed search (`discovery_start` 1791843199.6977184) found
SARAF again. Server replay confirms `source_claim` **0.371**, with the model also
selecting 0.371; the candidate is **not cached** and was saved as
`disc_2606_04135_f7b4e6db785fdb63`, `runnable: true`. The old
`disc_2606_04135_e3e2931687c224d2` with 0.366 remains separate historical data.
Reproduction was disabled by its daily gate, rather than claim eligibility.

The source-backed live claim exposed another false-negative protocol warning:
decoded model JSON still contained two backslashes before LaTeX commands and
escaped braces in the setup quote. Comparing the full original SARAF source
proved the overescaped quote did not match, while correcting only those escapes
matched the complete original sentence exactly.

`_judge_protocol` now permits that serialization repair **only** when the entire
repaired quote occurs in the source. Numbers, words, task matching and semantic
protocol rules stay unchanged. Invented setup statements remain unknown. Only
cache entries carrying this precise failed-quote reason and overescaped syntax
are reread; the cache version and unrelated entries remain unchanged. Existing
published candidate events and recordings are not rewritten.

163 focused Discovery, persistence and reproduction-API tests passed (the same
two known query-limit tests deselected). Normal, doubled and quadrupled escapes,
invented setup statements and affected-cache rereading have regression coverage.
No paid run or deployment was initiated. The already saved correct 0.371 claim
can be checked when the reproduction gate allows it; another discovery is not
required merely to start that existing claim.

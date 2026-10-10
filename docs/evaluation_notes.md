# Evaluation and research notes

Detailed evidence behind [Nullius](../README.md). Historical benchmark configurations are separate from the current product default. Counts here do not establish performance on unseen papers; method fidelity is not checked.

## Our own small bitter lesson

![Same eight units in two modes: 2 of 8 matches reading the paper only, 7 of 8 reading the paper and the authors' code, with the held-out paper and the limits](../docs/img/two-modes.svg)

**What we did for days.** Benchmark v3 and v4 each came after a failure analysis of the run before, and each added engineering to make the agent cleverer: better code extraction, repair after timeouts, Tavily error search, three code candidates chosen by a validation metric, checks for prediction-to-node mapping, a graph repair loop. Agreement with the reference verdicts stayed at **2 of 8** (v2: 2 / 2, v3: 2 / 2, v4: 1 / 2 without / with Tavily; [`docs/benchmark_history.md`](../docs/benchmark_history.md)). Around it we tried more, all with rules fixed before the run, and most missed their rule:

| What we tried | Outcome |
|---|---|
| Setup details from paper and README (role 4), DLinear and Shchur | rule missed; Shchur by 0.0004 (`results/role4_dlinear.md`, `role4_shchur.md`) |
| Tavily objection search before the experiment (role 3) | no protocol objection found (`results/role3_dlinear_stage1.md`) |
| v5 steps A and C (dossier fidelity, trial run) | not passed, no v5 (`results/v5_step_a.md`, `v5_step_c.md`) |
| An LLM checker for method fidelity, three calibrations | 5/10, 7/10, 8/11 correct, all below the threshold set before; axis 2 stays "not checked" (`results/calibration_fidelity_v1.md` to `_v3.md`) |
| "Re-run" mode that executes the authors' code | works on the two repositories it was built on, **0 of 4** on new ones, twice (`results/nachlauf_v1.md`); not in the product |

A post hoc analysis of all stored generations ([`docs/recall_analysis_3.md`](../docs/recall_analysis_3.md)) explained why: per-generation hit rates predicted **1.96 of 8**, so 2 of 8 was the system's equilibrium, not bad luck. A unit only matches when every decisive detail is right, and for at least five of the eight units a decisive detail **exists only in the authors' repository, not in the paper**: the learning-rate halving per epoch (DLinear, NLinear), the early-stopping rule (GAT), a tuned weight decay that sits in a pickled file (SGC). No amount of prompt cleverness can use a fact that never reaches the prompt.

**What changed the number.** We gave the code model the authors' code itself: a verbatim excerpt of the public repository (model file, helper functions, start script, configs) goes into the plan, code and repair prompts, and a fact sheet enforces the numeric settings found there. The code is read as text, **never executed**; the agent still writes its own script and our harness still measures it. Before paying for the full run we ran two small pre-registered gates. G0 was stopped incomplete (50 of 54 samples; on the development units the authors'-code arm hit the reference verdict in 8 samples against 4 for the paper-only product (16 finished samples each), the pre-registered rule asked for a lead of 5, so it counts as neither passed nor failed). G1, on the two GAT units, passed a low bar (`results/recall_cg_gate.md`, `results/recall_cg_gate_g1.md`). The full run, **recall cg v1** (rules in [`recall_cg_v1.md`](../benchmark/experiments/recall_cg_v1.md), committed before the first model call; results in [`results/recall_cg_v1.md`](../benchmark/experiments/results/recall_cg_v1.md)):

| Unit | Reference verdict | Paper only (v4, with Tavily) | Paper and authors' code (generation 0 / 1 / 2) |
|---|---|---|---|
| Kipf & Welling, GCN | reproduced | reproduced, match | reproduced / reproduced, **match** |
| Wu et al., SGC | reproduced | not reproduced | reproduced / reproduced, **match** |
| Zeng et al., DLinear | reproduced | partially | reproduced / reproduced, **match** |
| Zeng et al., NLinear | reproduced | not reproduced | reproduced / not reproduced / not reproduced, miss |
| Shchur et al., GCN | reproduced | reproduced, match | reproduced / reproduced, **match** |
| Shchur et al., GAT | reproduced | partially | partially / reproduced / reproduced, **match** |
| Veličković et al., GAT | reproduced | partially | setup failure / reproduced / reproduced, **match** |
| Veličković et al., DGI | partially | not reproduced | partially / reproduced / partially, **match** |
| **Development units matching** | | **2 of 8** | **7 of 8** (in-sample) |
| Li et al., RLinear (**held-out**) | reproduced | not in v4 | not reproduced / not reproduced, miss: **0 of 1** |

Placebo claims in the same run: **0 of 9 confirmed** (8 rejected, 1 without a verdict because that generation failed in setup). Cost: 5.49 $ gross for the 22 counted runs (7.45 $ including an aborted first attempt, when Token Factory removed Ultra mid-run). A live check of the product default on 9 Oct 2026 (Kipf & Welling, 3 seeds, Nebius and Token Factory): reproduced, **0.8213 ± 0.0021** against 0.815, 9.6 minutes, 0.139 $ gross ([run](../web/public/data/disc_1609_02907/20261009-115012/zeugnis.md)).

**The bitter lesson, our version.** Rich Sutton's essay [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019) makes one point that researchers keep relearning the hard way: "general methods that leverage computation are ultimately the most effective, and by a large margin", and the human knowledge we build into our systems helps in the short term but plateaus. Our small, related lesson is not the same one. We did not scale computation; we spent days of hand-built cleverness (prompts, repair rules, a checker) and gained little, while **more and better information** moved the count: *the missing details were in the authors' code, not in a cleverer prompt.* It cuts both ways: the held-out RLinear miss is again a detail that never reached the prompt (learning-rate halving per epoch, in the authors' `exp_main.py`; plus three input defects, [`docs/rlinear_cg_v1_analysis.md`](../docs/rlinear_cg_v1_analysis.md), post hoc and not proven by an ablation). We learned this on eight units, in-sample, with one held-out miss. It is an observation, not a proof.

**What this does not show.**
- **In-sample.** The fixes were designed after seeing these eight units. The only held-out paper missed in both generations (0.487 and 0.432 against a target of 0.366).
- **"Recall improved" is not claimed.** Our pre-registered reading rule required the held-out check as well, and it failed. The allowed statement is exactly the three numbers above (7 of 8 in-sample, 0 of 1 held-out, 0 of 9 placebos), and v4 (2 of 8) and this run are **never summed or netted** into one gain.
- **Generation variance is large.** Four of the eight units needed a third generation; for three of the seven matches (Shchur-GAT, Veličković-GAT, DGI) the unit verdict hangs on that one generation, and NLinear flipped from a match to a miss. Counted per single generation, 15 of 22 verdicts matched the reference (post hoc).
- **Different mode, models and rules.** This run used Nemotron 3 Super in every role (Ultra was removed from Token Factory on 2026-10-09), stored Tavily results in place of live calls, the error search switched off, placebos only in generation 0, an adaptive third generation and a higher cost cap. It is not comparable with v4 under the v2 rules, and it says nothing about Tavily's contribution.
- **Not "from the paper alone" any more.** The verdict now means "the target was reached by our rebuild of the method as paper **and code** define it, measured by our harness". Every certificate names its mode. Method fidelity is not checked.


## How we use Tavily

A normal run calls Tavily **live** at run time (`TAVILY_MODE=live`, the default). Every source is recorded with mode, timestamp, SHA-256 and length in the dossier, in `events.jsonl` and as `paper_text.json`, so a run stays traceable if a web page changes. The live and evidence views show a badge per Tavily call.

| Where | Endpoint | Purpose |
|---|---|---|
| **Discovery (live catalog)** | Search (github.com, arxiv.org, openreview.net) | finds new papers on supported benchmarks, reads protocol and results sections |
| Read claims (role 2) | arXiv API (Tavily as fallback) | LaTeX source is downloaded directly via arXiv API to extract tables with target values. Checked by quote and table column. |
| Material and setup details (roles 1, 4) | Search for the repository, Extract on the README, sandbox clone of chosen files | hyperparameters and data splits, each with a quote from paper, README or repository |
| Error search (role 5) | Search with the failing line | solution hint for the code repair |
| Community check (roles 3, 6) | Search, limited Crawl of repository issues | after the verdict: "What others found" and "Possible explanations", each with a quote. Verified findings that can be repeated become sandbox experiments (below). **Never changes the verdict** (`src/repro_agent/community.py`, `objection_experiments.py`) |

**Tavily findings that became experiments.** In a live DLinear run (8 Oct 2026, paper-only mode) Tavily's community search returned statements about the paper, for example that literature baselines in other papers may use different seeds and preprocessing, and a GitHub issue reporting an MSE of 0.426 against 0.386. Each statement that could be repeated ran as its own sandbox branch and was measured by the harness: five extra seeds gave 0.3793 ± 0.0015 ("seed dependence refuted"), and the issue's 0.426 was compared with our own 0.3793 (our measurement supports the paper's value, not the reported one). See the "Findings that became experiments" screenshot above and [`results/tavily_experiments_addendum_b.md`](../benchmark/experiments/results/tavily_experiments_addendum_b.md).

**A find we hoped for, and did not get.** For ETTh1 the harness can also re-score the same predictions with the last incomplete test batch dropped, as LTSF-style data loaders do (`drop_last=True`, batch size 128): all 2785 test windows **0.3793**, the last 97 skipped **0.3754**, paper **0.375**. The verdict still uses all windows. That note and its sources are a fixed list in the code, written from our own manual web research; **it is not a Tavily result.** We tried to let Tavily find the notice and turn it into this experiment by itself (pre-registered, three live runs, addenda A to C): the search did not return a page that names it, so the automatic path never ran live. It works on stored predictions and in tests. Findings become experiments when the search returns a measurable statement, not on demand.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/img/droplast-dark.jpg">
  <img src="../docs/img/droplast.jpg" alt="Evidence card 'A known evaluation setting': claimed 0.3750, all windows 0.3793, skipping 97 windows 0.3754">
</picture>

Tavily's effect on the benchmark is small (see below); its main role in the product is Discovery, the traceable sources and findings that turn into measurements. The Tavily key is scarce, so tests and the benchmark use stored copies (`--offline`, `TAVILY_MODE=fixture`; in v4 only role 5 was live, in the authors'-code run none).

**Not built:** Tavily Research for citable sources (role 7), possible explanations of a deviation as their own hypotheses (role 6), and the "re-run" mode that executes the authors' code.


## Benchmark

**Scope: small graph and time-series models that run on CPU in minutes.** Six papers, eight paper-claim units, each with and without Tavily, three fresh generations per unit and arm, plus a deliberately wrong **placebo** claim per run (for example 88 % on Cora). 48 runs, product configuration of that time (paper-only mode), stored Tavily results in both arms. Rules fixed before the first run ([`benchmark_v2.md`](../benchmark/experiments/benchmark_v2.md), v4 changes in [`benchmark_v4.md`](../benchmark/experiments/benchmark_v4.md)); results in [`results/benchmark_v4.md`](../benchmark/experiments/results/benchmark_v4.md), code `bf520c1` and `2b13b31`, numbers exported by `scripts/export_benchmark_json.py` (also shown on the Benchmark page of the web app).

**1. Precision first.** In benchmark v4, no placebo was reported as "reproduced": 0 of 48 in v4 and, with different code, 0 of 48 in each of v2 and v3. All 16 placebo majorities in v4 say "not reproduced". Every "reproduced" verdict for a real claim in v4 (7 of 7) matched our reference solution (post hoc, small sample).

**2. Primary metric** (pre-registered): per unit, the majority verdict of three generations against the reference verdict. With Tavily **2 of 8** units agree (Kipf-GCN, Shchur-GCN), without Tavily **1 of 8** (DLinear). The pre-registered expectation for v4 (more agreement than v3 in at least one arm) was **not met**.

**3. How to read it.** The errors are one-sided: In v4, Nullius often missed the rebuild and confirmed none of the planted false targets. A miss means "not reached with our setup" and usually points at the agent, not at the paper. Misses are mostly consistent near misses: the median distance to the target value is 0.017 with and 0.029 without Tavily (post hoc), "partially" is the most frequent miss. What improved from v2 to v4 is stability (no unit without a majority) and failures (1 of 48 real-claim verdicts without a measurement). Tavily makes no measurable difference at this sample size (2 vs 1 of 8).

| Unit (task) | Reference verdict (source) | Majority without Tavily | Majority with Tavily | Placebo detected (both arms) |
|---|---|---|---|---|
| Kipf & Welling 2017, GCN (Cora) | reproduced (R) | not reproduced | **reproduced** | yes |
| Wu et al. 2019, SGC (Cora) | reproduced (R) | partially | not reproduced | yes |
| Zeng et al. 2023, DLinear (ETTh1) | reproduced (R) | **reproduced** | partially | yes |
| Zeng et al. 2023, NLinear (ETTh1) | reproduced (R) | not reproduced | not reproduced | yes |
| Shchur et al. 2018, GCN, random splits (Cora) | reproduced (R, H) | partially | **reproduced** | yes |
| Shchur et al. 2018, GAT, random splits (Cora) | reproduced (R, H) | partially | partially | yes |
| Veličković et al. 2018, GAT (Cora) | reproduced (R) | partially | partially | yes |
| Veličković et al. 2019, DGI (Cora) | partially (R) | not reproduced | not reproduced | yes |
| **Units agreeing with the reference** | | **1 of 8** | **2 of 8** | **8 of 8 / 8 of 8** |

| | v2 | v3 | v4 |
|---|---|---|---|
| Units agreeing with the reference, without / with Tavily | 2 / 2 of 8 | 2 / 2 of 8 | 1 / 2 of 8 |
| Placebo verdicts "reproduced", of 48 | 0 | 0 | 0 |
| Placebo majorities "not reproduced", of 16 | 13 | 14 | 16 |

Each version ran different code; v2 and v3 in full: [`docs/benchmark_history.md`](../docs/benchmark_history.md). Every v4 miss is traced to its cause in [`failure_analysis_v4.md`](../benchmark/experiments/failure_analysis_v4.md) and [`results/failure_log_v4.md`](../benchmark/experiments/results/failure_log_v4.md). One v4 process died on a sandbox connection error and was restarted with unchanged code ([`benchmark_v4_incidents.md`](../benchmark/experiments/results/benchmark_v4_incidents.md)). The authors'-code run (recall cg v1) is a separate, smaller measurement with its own rules; its per-unit table is in [Our own small bitter lesson](#our-own-small-bitter-lesson).

**Limits, stated up front.**
- **Method fidelity is not checked.** The LLM checker failed its pre-registered threshold in three calibrations (5/10, 7/10, 8/11; [`calibration_fidelity_v1.md`](../benchmark/experiments/results/calibration_fidelity_v1.md), `_v2.md`, `_v3.md`). Axis 2 stays "not checked", every verdict is provisional, and the benchmark counts axis 1: "target value reached according to the harness". Only the graph ablation says something about method use, for graph models.
- **No real negative case.** Dacrema et al. (2019) was dropped (on MovieLens-1M the paper itself reproduces NeuMF) and Shwartz-Ziv & Armon was scoped and dropped ([`shwartz_ziv_scoping.md`](../benchmark/experiments/shwartz_ziv_scoping.md)). The placebo claims are constructed and do not replace a real negative case.
- **Reference verdicts.** (R) = our own reference solution run through the same harness, so not independent of our protocol choices; (H) = independent re-measurement in a critique paper (Shchur et al., GCN and GAT).
- **Small numbers.** Eight units are counts, not a significance test. Both arms get the same target value, so a difference comes from setup details and prompt length.
- The v4 evaluation covered Cora and ETTh1, CPU only; DGI was meant as a Tavily holdout and is no longer clean ([`tavily_v2_holdout.md`](../benchmark/experiments/tavily_v2_holdout.md)).

All numbers in this README and in the video come from production runs on Token Factory.

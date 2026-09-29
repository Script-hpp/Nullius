# Paper reproduction agent — the paper is the test suite

> Other agents fix code until the tests turn green. Here, the paper is the test suite.

Built for the Nebius × NVIDIA Global AI Hackathon (track: Coding and Agentic Engineering, plus Best Use of Tavily).
Licence: MIT.

**Status: work in progress.** Benchmark numbers below are placeholders (`TBD`) until the production runs on Nebius Token Factory are finished.
Everything else in this README is backed by a file in this repository; pointers are given in brackets.

## Problem and solution

Many machine-learning results are hard to reproduce. Rebuilding a paper costs days of broken code, missing hyperparameters and numbers that do not come out.

Give the agent an arXiv paper. It

1. extracts a few checkable claims with target values (Tavily Extract on the paper text, every number quoted verbatim and re-checked against the text and the table header),
2. collects setup details and community objections (Tavily Search / Extract / Crawl),
3. writes its own experiment code and runs it in sandbox branches,
4. lets an **independent harness** score the predictions, and
5. issues a reproduction certificate: a verdict per claim with measured mean, spread, sources and the branch tree as evidence.

The agent delivers *predictions*, never numbers. Verdicts are worded carefully ("not reproduced *with our setup*") and always list the attempts.

## Architecture

```
App (Python orchestrator)
 ├── LLM client ───▶ Nebius Token Factory (NVIDIA Nemotron); OpenRouter free endpoints only for prototyping
 ├── Sandbox ──────▶ Nebius Sandboxes (real branching, contree-sdk) | local Docker fallback
 ├── Tavily ───────▶ Search, Extract, Crawl (seven roles, see below)
 ├── Harness ──────▶ our own code, copied fresh into the branch after the agent has run
 └── Report ───────▶ certificate (Markdown) + dossier (JSON)
```

Nemotron is the brain, the sandbox is the lab, Tavily is the researcher, the orchestrator is the lab head. The model never executes code; the sandbox never thinks.

| Role | Task | Model (production) |
|---|---|---|
| `fast` | triage, table extraction | Nemotron 3.5 Lightning |
| `code` / `extract` | experiment code, extraction of setup details and of the method checklist | Nemotron 3 Super |
| `reasoning` | research plan | Nemotron 3 Ultra |

Tavily roles: (1) find material, (2) read claims from the paper, (3) collect objections, (4) fetch setup details, (5) look up error messages, (6) explain deviations, (7) cite sources. Every call must answer a question the agent could not answer otherwise. The Tavily key is scarce, so tests run on fixtures and the paper text is cached.

## How a verdict is produced

**Independent harness.** The agent writes `/output/predictions.csv` (`id,prediction`); nothing else counts. After the agent's run, the harness is copied fresh into the branch, loads the test labels itself from the original dataset, checks plausibility (complete unique ids, not a single class) and computes the metric with fixed, tested code.

**Leak protection** (recorded in the dossier under `isolation`): the training run has no network (`unshare -n`); a static code check rejects network access, downloads, process spawning and dynamic code; test labels never enter the agent's dataset.

**Two axes.** A verdict is not one number:

| Axis | Values | Source |
|---|---|---|
| 1. Result | reproduced / partially / not reproduced / agent fault suspected / not reproducible (setup) | mean over all seeds against a tolerance fixed before the run |
| 2. Method fidelity | yes / partially / no / not checked | method checklist, graph ablation |

Overall statement: *reproduced + method yes* = "paper reproduced"; *reproduced + method no/partially* = "target value reached with a different model"; *not reproduced + method yes* = "not reproduced"; *method not checked* = "provisional". The benchmark counts only the first and the third.

**Baselines.** The harness reports trivial baselines (majority class, no-graph logistic regression, repeat-last-value). A measured value that is worse than the best trivial baseline although the paper's value beats it is flagged "agent fault suspected" instead of blaming the paper.

**Graph ablation (GCN).** The agent's code runs a second time on the same data *without edges*. If accuracy does not drop, the graph was never used, however the code looks. Rule fixed in advance: drop ≥ 0.10 = "graph used", ≤ 0.02 = "graph not used" (axis 2 becomes "no"), otherwise unclear. [`benchmark/experiments/ablation_gcn.md`]

**Form check before training.** The agent's script first runs with `SMOKE=1` on a tiny subset; the output form is compared with the task's contract (for DLinear `(windows, 96, 7)`), so tensor-shape errors surface within seconds and the repair loop gets a precise message.

**Column check for target values.** The table header is parsed deterministically and the position of the number in the row is compared with the position the header dictates. Example (DLinear, Table 2): DLinear must be the third MSE/MAE pair, Repeat the last.

## Why hitting the target value proves nothing

In the DLinear experiment one fresh generation of the agent was judged "reproduced" with MSE 0.3765 (target 0.375, tolerance 0.01) and a standard deviation of exactly 0 over five seeds. The code was a closed-form ridge regression, not DLinear: no decomposition, no two linear branches. The number was right, the method was not. This is why the certificate has a second axis, and why "reproduced" alone is never reported as "method reproduced". [`benchmark/experiments/results/role4_dlinear.md`]

## Validating the harness itself

- **DLinear / ETTh1.** The harness' own repeat-last-value baseline gives MSE 1.2944 / MAE 0.7132 on all 2785 test windows; the paper reports 1.295 / 0.713 for `Repeat`. A human reference DLinear, re-measured through the pipeline (5 seeds, no LLM): MSE 0.4278 with a constant learning rate and 0.3809 (median 0.3751) with the per-epoch halving used in the original repository (paper: 0.375). [`results/etth1_harness_reference.txt`]
- **Cora, random splits (Shchur et al. 2018).** A human reference GCN with the paper's Appendix C configuration, run through the same pipeline on 30 random splits × 1 initialisation, reaches accuracy 0.8107 with a standard deviation of 0.0118 (paper: 81.5 ± 1.3 over 100 × 20 runs). The spread across runs matches the paper's. 90 runs (30 × 3) give 0.8111. [`benchmark/experiments/shchur_task.md`, `results/lcc_harness_reference_*.txt`]
- **Graph ablation** on ten hand-written GCN scripts (eight use the graph, two build it and never multiply with it): 9 of 10 classified as expected; the tenth (no self-loops, no edges → constant output) was "not measurable" by the rule defined beforehand. Thresholds were not adjusted. [`results/ablation_gcn_validation.md`]

## Experiments with rules fixed in advance

Every experiment has a file written and committed **before** the first run: hypothesis, setup, counting rules, success rule, limitations. Rules are not changed after seeing results, even when they become unreachable; results that contradict the hope are reported. Evaluation uses the median, not the mean (one broken generation dominates a mean).

| Experiment | Question | Result |
|---|---|---|
| Role 4 on DLinear (`role4_dlinear.md`) | Do setup details from paper + repo help? | **Primary rule clearly missed.** Near the target: arm without role 4 6 of 8, with role 4 1 of 8 (required: +3 for role 4). Mean MSE 0.5724 without vs 0.5635 with (required: −0.02). Post hoc: the details were complete but scattered over four repository files with contradictions; three of eight role-4 generations failed completely; the prompt was ~6× longer. |
| Method-fidelity checker calibration v1 (`calibration_fidelity.md`) | Can an LLM checker tell faithful from unfaithful code? | **Rule not met:** 5/10 correct, two yes↔no swaps. |
| Calibration v2, new codes, checker improved (`calibration_fidelity_v2.md`) | same | **Rule not met:** 7/10 (threshold 8/10), no swap; all errors were faithful codes rated too strictly. |
| Calibration v3, final (`calibration_fidelity_v3.md`) | same | **Rule not met:** 8/11 (threshold 9/11), no swap. |
| Role 4 on Shchur, paper appendix only (`role4_shchur.md`) | Does role 4 help when the settings sit in one place? | *Running. Result: TBD.* |

**Consequence of the three calibrations (fixed beforehand):** axis 2 ("method fidelity by the checker") stays **"not checked"** in the benchmark, and this is stated openly here. Verdicts are then "provisional" unless the deterministic graph ablation applies. The checker never rated a wrong code "yes" in the rule sets, but it also cannot see code that runs and whose result is discarded (a known limit of an existence check; line coverage cannot see it either, the ablation can).

## Benchmark

**Scope: small graph and time-series models that run on CPU in minutes.** A broad benchmark with shaky tasks would prove less than a narrow, well-documented one. Shwartz-Ziv & Armon (tabular data) was scoped and dropped on purpose: its splits come from other papers, its tuning budget (1000 HyperOpt steps per dataset) cannot be matched on CPU, and the metric is ambiguous, so a verdict would have measured guessed protocol details. [`benchmark/experiments/shwartz_ziv_scoping.md`]

Eight or more papers with known reproduction outcome, each run with and without Tavily. Target sentence for this section: "With Tavily, X of 10 verdicts agreed with human reproduction studies, without Tavily Y."

| Paper | Expectation | Task built | Result (with / without Tavily) |
|---|---|---|---|
| Kipf & Welling 2017, GCN (Cora) | reproduced | yes | TBD |
| Zeng et al. 2023, DLinear (ETTh1) | reproduced | yes | TBD |
| Shchur et al. 2018, Pitfalls of GNN Evaluation (Cora, random splits) | GCN best on random splits | yes (GCN number only) | TBD |
| Wu et al. 2019, SGC (Cora, Planetoid split) | reproduced, on par with GCN | yes (same Cora task) | TBD |
| Ferrari Dacrema et al. 2019, Are We Really Making Much Progress? | baselines beat neural models | no | TBD |

All numbers in this README and in the video come from production runs on Token Factory; prototype runs on free endpoints are never published.

## Cost tracking, reconciled with the Nebius bill

Every model call is logged (`logs/costs.jsonl`): provider, base URL, model, input/output tokens taken from the API's `usage` field, cache hit, finish reason, net cost. Compared with the Nebius console (29 Sep 2026): Ultra, Super and Lightning token counts agree with the console within its rounding (0.01 M tokens); net total 1.256 $ in the log vs 1.26 $ in the console, gross 1.495 $ vs 1.49 $ (19 % VAT). The log remains an estimate; the console is authoritative, and the price table in the code matches the official prices.
Safeguards: a per-run limit (gross), a global budget with a cut-off date (warning at 80 %, abort at 95 % before the first call), retries with exponential back-off, and a hard stop on billing or authentication errors (no retries, no repair loop). Truncated model answers are never cached.

## Limits and open points

- CPU only, claims that run in under ten minutes; no LLM papers, no PDF image analysis (arXiv HTML/PDF text only).
- Mode "rebuild" only: the agent writes its own code and takes only cited details from paper and repository; running the authors' code ("re-run") tests something different and is not built.
- Method fidelity by the LLM checker is not calibrated (see above). The graph ablation exists for GCN only; DLinear would need a decomposition ablation.
- Hand-written calibration and ablation codes can differ from real agent code; labels were written by the authors of the codes.
- Shwartz-Ziv & Armon was dropped after scoping (see Benchmark). GAT (Veličković et al., 2018) is a candidate if time remains; agents tend to make more mistakes implementing it.
- The Shchur task checks the GCN number, not the ranking of models; 30 random splits × 1 initialisation instead of 100 × 20.
- Role 3 (collecting objections) and role 4 v2 (effective configuration, complete bundle) are designed but not evaluated.
- Nebius sandbox rate limits: 20 parallel branches ran without errors; higher was not tested.
- Reference numbers of the human reference solutions are pipeline self-tests, not reproduction verdicts.

## Related work

- **PaperBench** ([Starace et al., 2025](https://arxiv.org/abs/2504.01848)): agents replicate 20 ICML 2024 Spotlight and Oral papers from scratch and are graded against hierarchical rubrics (8,316 gradable sub-tasks) co-developed with the papers' authors, using an LLM judge.
- **CORE-Bench** ([Siegel et al., 2024](https://arxiv.org/abs/2409.11363)): 270 tasks from 90 papers in computer science, social science and medicine; agents run the code and data provided with a paper and answer questions about the results.
- **Difference:** our agent is judged against the *paper's claims*, not a rubric; the score comes from an independent harness the agent cannot touch; a verdict can say "not reproduced", and negative findings are a first-class output. We also test our own components (harness, checker, ablation) with rules fixed beforehand and report the failures.

## Usage

```bash
cp .env.example .env            # add NEBIUS_API_KEY, NEBIUS_PROJECT_ID, TAVILY_API_KEY
uv sync
LLM_PROVIDER=tokenfactory uv run repro-agent kipf_gcn_2017
uv run python -m pytest -m "not integration"   # unit tests, no network
uv run python scripts/cost_report.py            # cost log vs console
```

Configuration lives in `.env` (`LLM_PROVIDER`, `SANDBOX_BACKEND`, budgets, VAT rate). Paper definitions and targets are in `benchmark/papers.yaml`; run results are written to `runs/<paper>/<timestamp>/` (`dossier.json`, `zeugnis.md`).

## Inspiration

*Nullius in verba*: take nobody's word for it. Many machine learning results are hard to reproduce. Rebuilding a paper costs days of broken code, missing hyperparameters and numbers that do not come out, and whether a claim holds is usually known only after someone has paid that price. Coding agents today fix code until the tests pass. We wanted an agent for which **the paper is the test suite**.

## What it does

You give Nullius a machine learning paper (an arXiv link or a catalog entry). It then:

1. reads the paper with **Tavily** Extract and pulls out checkable claims with target values; every number is quoted verbatim and re-checked against the text,
2. collects setup details (hyperparameters, data splits) from the paper, the README and the repository, each with a verbatim quote and its source,
3. lets **NVIDIA Nemotron** plan and write its own experiment code and runs it in **Nebius Sandbox** branches, one branch per seed, with no network during training,
4. lets an **independent harness** score the result: the agent delivers only `predictions.csv`; the harness loads the test labels itself and computes the metric with fixed, tested code,
5. issues a certificate per claim: *target reached*, *partially reproduced* or *not reproduced with our setup*, with the measured mean, the spread over seeds, the sources and the branch tree as evidence.

The product has five pages: Overview, **Discover** (Tavily finds new papers on supported benchmarks), **Catalog**, **Live run** (plan, code, branches, measurements and repairs stream in live) and **Evidence** (the certificate with a step-by-step replay).

**Honest scope.** Nullius checks papers that evaluate on benchmarks it already supports: Cora (fixed and random splits) and ETTh1 (horizon 96). CPU only, claims that run in minutes. "Reproduced" means the target value was reached according to the harness; **method fidelity is not checked**.

## How we built it

- **Orchestrator** (Python): per claim, Tavily reads target value and setup details, Nemotron plans and writes code, the sandbox runs it, the harness scores it, checks run, the code is repaired if needed, then all seeds run in parallel branches.
- **NVIDIA Nemotron on Nebius Token Factory**, one OpenAI-compatible endpoint: Nemotron 3.5 Lightning for table and target-value extraction, Nemotron 3 Super for writing and repairing code and for setup details, Nemotron 3 Ultra for the research plan. Every quote a model returns is re-checked against the source and dropped if it does not occur there.
- **Nebius Sandboxes** (`contree-sdk`): the base image is built once as a snapshot; every seed, repair attempt and ablation is a branch of it. Rollback of a failed attempt is free.
- **Tavily** Search and Extract at run time for discovery of new papers (the live catalog), claims, setup details and error search; after the verdict a community check reports what others found, with quotes, and never changes the verdict. Every source is stored with mode, timestamp and SHA-256.
- **Checks without an LLM:** a smoke run on a tiny subset, trivial baselines, a graph ablation, a test for training on validation labels, an edge-direction probe, and a detector for a known data-loader bug (`drop_last`) in time-series papers.
- **Product:** FastAPI with a queue, daily limits, an access code and Server-Sent Events; a React app for the five pages.

## Challenges we ran into

- **Keeping the agent honest.** An agent that writes code and reports numbers can fool itself. So the agent delivers predictions only; harness, labels and metric are ours. In our benchmark one generated program trained on the validation labels and won the candidate selection with a validation accuracy of up to 1.0. We added a check that catches this.
- **Telling a paper fault from an agent fault.** A low number can mean the paper does not hold or the generated code is wrong. Baselines, the form check and the ablation separate the cases; a result below a trivial baseline is reported as "agent fault suspected", not as a finding about the paper.
- **The web can mislead.** A README tip ("it is often helpful to standardise the features") was followed faithfully by the agent and cost about eight accuracy points on one paper. We verified the cause with a counter-experiment and now mark such statements as optional advice.
- **Method fidelity.** We built an LLM checker for "did the code implement the method" and calibrated it three times. It missed its pre-registered threshold each time (5/10, 7/10, 8/11). We stopped and report that axis as "not checked".
- **Reasoning models that think inside the answer** used the whole token budget on a simple extraction. We split inputs into small sections and never cache truncated answers.

## Accomplishments that we're proud of

- A complete loop from arXiv paper to certificate, watchable live.
- **A pre-registered benchmark, with the failures reported.** Six papers, eight paper-claim units, each with and without Tavily, three fresh generations per unit, plus a deliberately wrong "placebo" claim in every run as negative control. Rules were fixed in the repository before the first run. Result of the latest version (v4, 48 runs): **none of the 48 placebo verdicts was "reproduced"** (also 0 of 48 in v2 and v3, runs with different code), and all 7 "reproduced" verdicts for real claims agreed with our reference solution (post hoc, small sample). Nullius is a high-precision checker. Recall is low: with Tavily **2 of 8** units matched the reference verdict, without Tavily **1 of 8**, mostly near misses. We publish it as it is, together with a failure analysis that names the cause run by run. Method fidelity was not checked.
- **A finding about a published number** (a manual check by the team, not an agent result): for RLinear on ETTh1, the authors' code prints an MSE of 0.366, exactly the value in the paper, but it scores only 2688 of the 2785 test windows because the test loader drops the last batch. On all windows the same model reaches 0.370. Our harness reports both.
- Negative results are first-class output: three failed calibrations and one improvement step that missed its own pre-registered goal are documented, not hidden.

## What we learned

- Free-form agent output needs a verifier the agent cannot touch. Verdict and number must not come from the same component.
- Most misses were not coding errors. The decisive detail was missing from the collected setup details, or it was quoted correctly but belonged to a different model or setting. Of the details that were collected, the code used 84 %.
- The variation between two generations of the code is larger than the variation between seeds. One run of an agent proves little; we count majorities over three generations.
- A narrow, well-documented benchmark says more than a broad one with shaky tasks. Eight units are counts with caveats, not a significance test.
- Sandbox branching fits this problem: seeds run in parallel from one snapshot, and every attempt stays inspectable.

## What's next for Nullius in verba

- **A second mode, "re-run":** run the authors' own code in the sandbox and measure it with the same harness, next to the agent's rebuild. Together the two answer a new question: does the number hold, and does the paper describe enough to get it? A manual pre-study on two repositories passed its pre-registered criteria; the agent version is in progress.
- A method-fidelity check that passes calibration.
- More datasets and a real negative case in the benchmark.
- Notifying paper authors with the certificate (the README badge for authors already exists).

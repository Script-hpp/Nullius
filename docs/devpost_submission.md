# Devpost submission text (draft)

Draft of every text field of the submission form. English. `README.md` is the reference: where this text and the README differ, the README wins. Every claim points to a file in the repository. Benchmark numbers are from Benchmark v4 (`benchmark/experiments/results/benchmark_v4.md`, pre-registration `benchmark/experiments/benchmark_v2.md` and `benchmark_v4.md`); values marked `TBD` depend on the deployment (demo URL, video, mirror). Deadline: 30 Oct 2026, 10:00 PDT (18:00 German time).

## Project name

Nullius

## Tagline (one sentence)

Other agents fix code until the tests pass; Nullius takes one claim from a machine learning paper, rebuilds the experiment in Nebius Sandboxes and lets an independent harness measure whether the number holds. A high-precision checker: it never reported a deliberately wrong claim as reproduced.

## Track and prizes

- Track: Coding and Agentic Engineering.
- Bonus prize: Best Use of Tavily (Tavily is called live at run time; the calls are visible in the event log and on the evidence page, `README.md` section "How we use Tavily").
- Feedback prize: the feedback field points to [`docs/nebius_feedback.md`](nebius_feedback.md).

## What it does

*Nullius in verba*: take nobody's word for it. Many machine learning results are hard to reproduce. Rebuilding a paper costs days, and whether a claim holds is usually known only after someone has paid that price. Nullius is for reviewers who want a quick check of a headline number, PhD students who want to build on a paper, and ML teams that depend on published results.

You give Nullius a paper (an arXiv link or an entry of the catalog). It then:

1. reads the paper with Tavily Extract and extracts checkable claims with target values; each number is quoted verbatim and re-checked against the text and the table header,
2. collects setup details (hyperparameters, data splits) from the paper, the README and the repository, each with a verbatim quote,
3. writes its own experiment code and runs it in Nebius Sandbox branches, one branch per seed,
4. lets an **independent harness** score the predictions: the agent delivers only `predictions.csv`, the harness loads the test labels itself and computes the metric with fixed, tested code,
5. issues a certificate: a verdict per claim with measured mean, spread over seeds, sources and the branch tree as evidence.

**What the benchmark shows** (v4, 48 runs on Token Factory, `README.md` "Benchmark"): Nullius is a high-precision checker. When it reports a target as reached, it was (7 of 7 "reproduced" verdicts for real claims agreed with our reference solution; post hoc, small sample), and none of 48 deliberately wrong claims was reported as reproduced (pre-registered negative control). It often does not reach the target: 2 of 8 paper-claim units with Tavily, 1 of 8 without; most misses are near misses ("partially"). Method fidelity was not checked.

The product has five pages: Overview, **Discover** (Tavily finds new papers on supported benchmarks and proposes claims), **Catalog** (candidates that already ran once through the full loop), **Live run** (the loop streams over Server-Sent Events: Tavily findings, plan, code, branches, measurements, repairs) and **Evidence** (the certificate, with a step-by-step replay). If a live run cannot start, the UI says so and offers the last finished run as a replay (`README.md`, "Product tour").

Verdicts are worded carefully: "not reproduced *with our setup*", always with the attempts. A verdict has two axes; axis 2 (method fidelity) is **not checked** in this submission, so "reproduced" means "target value reached according to the harness", not "method reproduced" (`README.md`, "How it works" and "Benchmark").

**Honest scope.** Nullius checks papers that evaluate on benchmarks it already supports: Cora (Planetoid split), Cora with random splits, and ETTh1 (horizon 96). A new dataset needs a one-time task and harness; new papers on existing datasets work without one. CPU only, claims that run in minutes (`README.md`, "Limitations and roadmap").

## How we built it

- **Orchestrator:** a Python loop (`src/repro_agent/loop.py`) per claim: Tavily reads target value and setup details, Nemotron plans and writes code, the sandbox runs seed 0 in a branch, the harness scores it, checks run, the code is repaired if needed, then the remaining seeds run in parallel branches with exactly that code. The verdict rests on the mean over all seeds against a tolerance fixed before the run.
- **NVIDIA Nemotron on Nebius Token Factory** through one OpenAI-compatible endpoint (`src/repro_agent/llm.py`): Nemotron 3.5 Lightning (table and target-value extraction), Nemotron 3 Super 120B-A12B (writes and repairs the experiment code, extracts setup details), Nemotron 3 Ultra 550B-A55B (research plan). Every quote a model returns is re-checked against the source text and dropped if it does not occur there.
- **Nebius Sandboxes** (`contree-sdk`, `src/repro_agent/sandbox/nebius.py`): the base image (Python, numpy, scipy, torch) is built once as a snapshot; every seed, repair attempt and graph ablation is a branch of it. The agent's training run has no network (`unshare -n`), a static code check rejects network access, downloads and process spawning, and the harness is copied in fresh afterwards.
- **Tavily** (`src/repro_agent/tavily_tools.py`, `discovery.py`, `community.py`): Search and Extract at run time for discovery of new papers (the live catalog), claims, setup details and error search; after the verdict a community check reports what others found and possible explanations, each with a quote, and never changes the verdict. Every source is recorded with mode, timestamp and SHA-256 in the dossier and in `events.jsonl`.
- **Checks and repair loops:** a smoke run on a tiny subset (`SMOKE=1`), trivial baselines (a result below the best trivial baseline is flagged "agent fault suspected"), a graph ablation for graph models (validated 9/10, `benchmark/experiments/results/ablation_gcn_validation.md`), a plan validator, a crash repair from the end of stderr, and a deterministic table-column check.
- **Cost control:** per-run limit, global budget with a cut-off, retries with back-off, hard stop on billing or authentication errors (`CLAUDE.md`, section "Budget und Kostenkontrolle"; `src/repro_agent/llm.py`).
- **Product:** FastAPI (`src/repro_agent/api.py`: queue with one job at a time, daily limits, access code, SSE) and a React app (`web/`). Container and hosting: `Dockerfile`, `deploy/`, `docs/deploy.md`.

Where Token Factory helped (stated only where the repository shows it, `README.md` section "Where Nebius Token Factory helped"): one endpoint for all three Nemotron models; token counts in the API's `usage` field agreed with the console within its rounding (net 1.256 $ in our log, 1.26 $ in the console on 29 Sep 2026); catalog runs cost between 0.025 $ and 0.093 $ per paper (gross, model cost only, Tavily excluded).

## Challenges

- **Keeping the agent honest.** An agent that writes code and reports numbers can fool itself or the user. The agent therefore delivers predictions only; the harness, the labels and the metric are ours. Leak protection has three layers (no network in training, static code check, labels never in the agent's dataset).
- **Telling a paper fault from an agent fault.** A low number can mean the paper does not hold or the generated code is wrong. Trivial baselines, the form check and the graph ablation separate the cases; a result below a trivial baseline is reported as "agent fault suspected", not as a finding about the paper.
- **Method fidelity.** We built an LLM checker for "did the code implement the method" and calibrated it three times. It failed its pre-registered thresholds each time (5/10, 7/10, 8/11; `benchmark/experiments/results/calibration_fidelity_v1.md`, `_v2.md`, `_v3.md`). We stopped calibrating and report axis 2 as "not checked".
- **Reasoning models that think inside the answer.** Nemotron 3.5 Lightning used its whole token budget on reasoning for a simple extraction and returned no JSON (`docs/nebius_feedback.md`). We split inputs into small sections, raised `max_tokens`, never cache truncated answers and use Super for extraction.
- **Parallel sandbox load.** Status polling of long operations hit the SDK's 10 s HTTP timeout under parallel load and ended benchmark processes; workaround and suggestions are in `docs/nebius_feedback.md`.
- **Infrastructure reliability and cost.** Retries with back-off, billing-error detection that stops instead of retrying, and a budget counter that reconciles with the console.

## Accomplishments that we're proud of

- A complete loop from arXiv paper to certificate: Tavily-read claims, Nemotron-written code, Sandbox branches, an independent harness, a two-axis verdict, all watchable live.
- **Precision, measured against a negative control.** Benchmark v4: six papers, eight paper-claim units, each with and without Tavily, three fresh generations per unit and arm, plus a deliberately wrong "placebo" claim per run (`benchmark/experiments/benchmark_v2.md`, `benchmark_v4.md`, fixed before the runs; evaluation `scripts/eval_benchmark_v2.py`). **0 of 48** placebo verdicts were "reproduced" (also 0 of 48 in each of v2 and v3, runs with different code) and all 16 placebo majorities were "not reproduced" (pre-registered). **7 of 7** "reproduced" verdicts for real claims agreed with the reference (post hoc, small sample). Recall is low and reported as it is: the majority verdict matched the reference for **2 of 8** units with Tavily and **1 of 8** without (pre-registered primary metric; our expectation of an improvement over v3 was not met). Method fidelity was not checked (`benchmark/experiments/results/benchmark_v4.md`).
- Cheap checks: between 0.025 $ and 0.093 $ of model cost per catalog paper (`README.md`).
- Every Tavily call is a run-time call that answers a question the agent could not answer otherwise, and every source is stored with a hash so a run stays traceable if the web page changes.
- Negative results are first-class output: three failed calibrations of our own checker and a v4 that missed its own pre-registered goal are documented in the repository, not hidden (`docs/benchmark_history.md`).

## What we learned

- Free-form agent output needs a verifier the agent cannot touch; verdicts and numbers must not come from the same component.
- Reasoning models need small tasks and large token limits for extraction, and truncated answers must not be cached.
- A narrow, well-documented benchmark proves more than a broad one with shaky tasks; eight units are counts with caveats, not a significance test (`README.md`, "Benchmark").
- Reference verdicts from our own human reference solutions are not independent of our protocol choices; only Shchur et al. give an independent re-measurement for two units (`benchmark/experiments/benchmark_v2.md`).
- Sandbox branching suits this problem: rollback of a failed attempt is free, and seeds run in parallel from one snapshot.

## What's next

(`README.md`, "Limitations and roadmap"; `docs/idea_author_notification.md`)

- Notify paper authors with the certificate (idea only, not built).
- A method-fidelity checker that passes calibration, and ablations for non-graph models (DLinear, NLinear).
- More tasks and datasets; a real negative case for the benchmark (Dacrema et al. was considered and dropped because MovieLens-1M gives no negative).
- The paused Tavily v2: objection search, bundled setup details, explanations of deviations as their own branches.

## Built with

Python 3.11, FastAPI, uvicorn, httpx, PyYAML, NVIDIA Nemotron 3.5 Lightning, NVIDIA Nemotron 3 Super 120B-A12B, NVIDIA Nemotron 3 Ultra 550B-A55B, Nebius Token Factory, Nebius Sandboxes (`contree-sdk`), Tavily (Search, Extract), React 19, Vite, Tailwind CSS, Recharts, React Flow (`@xyflow/react`), Framer Motion, Lucide icons, PyTorch, NumPy, SciPy, Docker, Server-Sent Events, MIT licence.

Sources of the list: `pyproject.toml`, `web/package.json`, `README.md` (base image: Python, numpy, scipy, torch).

## Links

| Field | Value |
|---|---|
| Repository | <https://github.com/Script-hpp/Nullius> (`README.md`) |
| Licence | MIT, `LICENSE`; must appear in the GitHub "About" box (`CLAUDE.md`, "Hackathon-Regeln"). `<!-- REVIEW: check after the push that GitHub shows the licence -->` |
| Demo URL | `TBD` (deployment open: `docs/deploy.md`; must stay reachable and free to test until 15 Dec 2026) |
| Video | `TBD` (YouTube, public, under 3 minutes; script: `docs/video_script.md`) |
| Mirror (fallback) | `TBD` (static GitHub Pages mirror, replay only, `docs/deploy.md`) |

## Testing instructions for the judges

Demo URL: `TBD`. Access code for live runs: `TBD` (the pages Overview, Catalog, Evidence and the replay work without a code; only starting a live run or a discovery needs it, `README.md`, "Product tour"). <!-- REVIEW: reviewer decided on an access code (README, CLAUDE.md strategy); the code stays TBD; user decides finally. docs/product_plan.md (29 Sep) planned an open demo; the Devpost rules accept credentials in the test notes. -->

What to try in five minutes:

1. **Overview** (`#overview`): read the scope statement and open a finished certificate (one card per paper claim, earlier runs fold out below it).
2. **Evidence** (`#/run/<id>`): on the left, what Tavily found and the verified target value with its quote; on the right, what the sandbox measured (mean, spread, seeds, tolerance); below, the branch tree; use the replay to step through the event log. The badge next to each Tavily call shows whether it was live.
3. **Catalog** (`#/catalog`): candidates with claimed and measured value. Enter the access code and start a **quick check** (3 seeds). One run at a time and daily limits apply; if another run is active, try again later. Expect a few minutes; the page shows the typical duration (from the timestamps of the exported runs) and links the pre-computed result.
4. **Live run** (`#/live/<job>`): watch Tavily calls, plan, code, branches and measurements arrive. If the run cannot start, the page offers the last finished run as a replay.
5. **Discover** (`#/discover`): with the access code, let Tavily search for new papers on Cora or ETTh1 (real Tavily calls, daily limit).

Notes for reading results:

- The green badge reads "Target reached": the target value was reached according to the harness; **method fidelity is not checked**. Runs of the same claim are grouped under one card on the Overview.
- "Agent fault suspected" means the result is far below a trivial baseline; it says nothing about the paper.
- "Not reproducible (setup)" means our setup could not produce a valid run within the attempt limit.
- Overall statements in live runs are "provisional" because axis 2 is not checked.
- Only supported datasets work: Cora (Planetoid split), Cora with random splits, ETTh1.

Run the repository locally: `README.md`, section "Setup and usage" (needs a Token Factory key, a Nebius Sandboxes project and a Tavily key).

## Feedback field (Token Factory, AI Cloud, NVIDIA tools and models)

See [`docs/nebius_feedback.md`](nebius_feedback.md). Paste its content into the form field; nothing is duplicated here.

## Differences and conflicts with `README.md` to report

1. **Demo access:** README: live runs need an access code. `docs/product_plan.md` (29 Sep): demo open for all with a hard budget. This text follows the README; the test notes contain the code (`TBD`).
2. **Benchmark numbers:** filled from `benchmark/experiments/results/benchmark_v4.md` (v4 is the main result; v2 and v3 are in `docs/benchmark_history.md`). The README and this text give the same v4 numbers. `CLAUDE.md` still describes the v2 pre-registration as "Stand"; the reviewer updates it.
3. **Nebius Compute VM as further Nebius service:** `docs/deploy.md` recommends Coolify. If the demo runs on Coolify, "Built with" and the "other Nebius services" statement name only Token Factory and Sandboxes. No conflict as long as that holds.
4. **Speed comparison with OpenRouter free endpoints:** no measured comparison exists; neither document makes a speed claim.
5. **Screenshots:** `docs/img/*.jpg` come from an older UI build (for example the Overview screenshot still shows the old "VERIFIED" label and the live screenshot an older sidebar). The README uses only `live-running.jpg`, `evidence.jpg`; retake all six before submission.
6. **Roles 3 and 6 of Tavily:** the previous README said they are not built; the community check (`src/repro_agent/community.py`, report only) exists, the README now says so. Tavily Research (role 7) and objections as their own experiments are not built.

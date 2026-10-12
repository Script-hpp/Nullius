# Devpost story — updated draft, 10 Oct 2026

English submission narrative. Same results and limits as [video_script.md](video_script.md) and [devpost_submission.md](devpost_submission.md). Unpublished draft; no user-test results claimed.

## Inspiration

*Nullius in verba*: take nobody's word for it. Coding agents fix code until tests pass. We wanted an agent whose test is a reported scientific claim, with evidence a researcher can inspect before building on a paper.

## What it does

Before a researcher or ML team builds on a machine learning paper, Nullius helps check whether its reported number can be reached. **Here, the paper is the test suite:** a claim becomes the acceptance criterion for an independently measured experiment.

Give Nullius an arXiv link or open a catalog entry. Tavily reads claims and setup details at run time. **The default mode rebuilds from the paper and the authors' code:** NVIDIA Nemotron on Nebius Token Factory reads repository files as reference material, then writes its own experiment. The authors' program is never executed. If reference code is unavailable, Nullius falls back to a paper rebuild and displays the actual mode.

Nebius Sandboxes execute the generated code in branches. The agent supplies predictions; an independent harness loads the test labels and computes the metric. A certificate records the claim, measured mean, spread across seeds, tolerance, sources, program versions and branches. The Evidence page offers a recorded replay and a browser print preview that can be saved as a **PDF certificate**. The PDF uses the stored result, not the What-if slider.

Use that evidence to decide what to investigate before building on a paper. We have not measured user time savings or completed a user study.

## Results and their scope

Three later catalog checks reached their fixed targets on ETTh1, forecast horizon 96, three seeds each, tolerance ±0.01 MSE. MSE is prediction error; lower is better. **All three are preliminary: method fidelity is not checked.**

| Paper | Claimed MSE | Measured mean ± standard deviation | Gross inference cost |
|---|---:|---:|---:|
| TSMixer (Google) | 0.361 | 0.3624 ± 0.0003 | $0.2354 |
| SparseTSF | 0.359 | 0.3653 ± 0.0034 | $0.2035 |
| GLinear | 0.3820 | 0.3796 ± 0.0003 | $0.2047 |

Evidence: `web/public/data/disc_2303_06053/20261009-201230/`, `disc_2405_00946/20261009-220822/`, and `disc_2501_01087/20261010-161916/`; dossiers contain measurements, event logs contain inference costs. Costs include configured VAT and exclude Tavily and sandbox costs. These examples were selected manually after development, outside the eight development units. They are not a blind test or a new general success rate. GLinear was already present in earlier search results.

GLinear's source was checked manually in paper v1 Table II before the run. Automatic target extraction returned no JSON, and automatic column assignment remained unchecked; the certificate keeps these warnings. TSMixer also retains protocol and table-column warnings. Reaching the target does not establish that the authors' method or exact training configuration was reproduced. See [paper_candidates_20261010.md](paper_candidates_20261010.md) and the certificates for details.

**Author-code development evaluation:** 7 of 8 development units matched the reference verdict, **in-sample**, after fixes were designed using those units. This counts agreement with reference verdicts, including a partial verdict, not seven fully reproduced papers. The separate **RLinear held-out check matched 0 of 1**. **0 of 9** planted controls were confirmed; one had no verdict. We do not combine these counts or claim a general recall improvement. Source: `benchmark/experiments/results/recall_cg_v1.md`. This evaluation used stored source fixtures, so it does not measure a Tavily accuracy gain.

**Earlier paper-only benchmark v4:** six papers, eight paper-claim units, 48 runs, three generations per unit and arm. With Tavily, majority verdicts matched the reference on 2 of 8 units; without Tavily, 1 of 8. **0 of 48 planted false targets were confirmed.** All 7 target-reached verdicts for real claims agreed with our reference solution (post hoc, small sample). This is evidence within that evaluation, not a universal precision guarantee. v4 used a different mode and historical model configuration; do not subtract its 2/8 from the development 7/8 or pool its controls with the later evaluation. Sources: `benchmark/experiments/results/benchmark_v4.md`, `docs/benchmark_history.md`.

Failed checks remain visible: SL-GCN missed its target, HaKAN failed in setup, and Acacia has no completed catalog verdict. A miss can reflect an agent or setup weakness; it is not a finding that a paper is wrong.

**Scope:** Cora (fixed and random splits), CiteSeer (Planetoid split), and ETTh1 (horizon 96), CPU only. Other datasets require a validated harness. “Target reached” means the numerical target was reached with our setup. Method fidelity is not checked, so overall conclusions stay provisional.

## How we built it

- **NVIDIA Nemotron on Nebius Token Factory:** Super plans, writes and repairs experiments and extracts setup details; Lightning extracts target values. Benchmark v2–v4 used Ultra for planning before its removal on 9 Oct; current runs use Super. Model-returned quotations are checked against source text; warnings remain when validation cannot complete.
- **Nebius Sandboxes:** snapshot branching for attempts and seeds, network disabled during training, static code checks, fresh harness and independently loaded test labels.
- **Tavily Search and Extract:** discovery, paper text, setup and error research at run time. Sources carry mode, timestamp and SHA-256. After the verdict, community findings and possible explanations are reported separately; measurable findings may become additional experiments, without altering the original verdict. We do not claim Tavily improves reproduction accuracy.
- **Python orchestrator and FastAPI, React product:** Overview, Discover, Catalog, Live run and Evidence; recorded checks remain accessible when live admission is paused. The catalog includes successful checks, failed checks and unchecked candidates.
- **Verification:** smoke checks, baselines, graph ablations and recorded repairs. The harness computes results from predictions; the agent never grades itself.
- **Cost controls:** logged inference costs, admission floors and daily/per-IP limits exist. The 10 Oct production fix added shared job budgets, durable model-request and Tavily-credit reservations, and cooperative timeout cancellation, verified with simulated providers. Actual provider billing, sandbox/hosting costs and external account usage remain outside that proof (`docs/cost_policy_review.md`).

## Challenges and what we learned

Separating agent errors from paper results matters more than a confident verdict. A result below a trivial baseline is flagged as an agent fault suspicion. Validation-label leakage, misleading repository advice and missing setup details required checks and documented counter-experiments.

Our method-fidelity checker missed three calibration thresholds (5/10, 7/10, 8/11). We stopped and label that dimension “not checked”. We also publish failed improvement steps and the separate held-out miss. Small, selected examples are useful demonstrations, but they do not establish performance on unseen papers.

Sandbox branches preserve failed attempts and let seeds run from the same snapshot. Stored sources, exact code and independent measurements make a certificate inspectable even after a web page changes.

## What's next

One person will test the finished product. We will record that person's observations before deciding further live-demo steps or author outreach. No authors have been contacted for this phase, and no email drafting button is part of the product. TimeBase, then SOFTS, are future paper candidates; **no new paid paper runs are being started**.

Longer-term work includes a calibrated method-fidelity check, more validated datasets and source findings tested as their own experiments. Those are plans, not current capabilities.

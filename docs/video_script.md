# Demo video — revision 9, 10 Oct 2026

Recording brief for the new cut. Historical brief: [video_script_v8_archive.md](video_script_v8_archive.md). Stored production evidence only; no new model, Tavily or sandbox calls. The previous film is preserved in the reviewer workspace.

Target length: **2:30**, English narration and readable English captions. Keep the warm paper, serif claims, precise measurement typography and restrained movement of the existing film. One result or action per shot. Mark app views as recorded and edits as shortened waiting time. Do not present a replay as a fresh live run.

## Cut and narration

| Time | Screen | English narration |
|---|---|---|
| 0:00–0:15 | Claim becomes acceptance criterion; central motto | Before your team builds on a machine learning paper, can you reach its reported result? Nullius turns a paper's claim into a test. Here, the paper is the test suite. |
| 0:15–0:27 | Current homepage → GLinear evidence | Open a finished check. See what the paper claims, what Nullius measured, and the evidence behind the result. This is a recorded run. |
| 0:27–0:40 | GLinear source and author repository | Tavily finds papers and setup details at run time. In the default mode, Nemotron on Nebius Token Factory reads the paper and the authors' code as reference material. |
| 0:40–0:52 | GLinear generated code | It writes its own experiment. The authors' program is never executed. If their code is unavailable, the product falls back to rebuilding from the paper and says so. |
| 0:52–1:05 | Actual GLinear repair events → seed measurements | Nebius Sandboxes run the experiment in branches. This stored run includes repairs. The agent supplies predictions; our independent checker measures them against a target fixed before the run. |
| 1:05–1:23 | TSMixer, SparseTSF, GLinear; one at a time | Three later catalog checks reached their targets within the fixed tolerance: TSMixer, SparseTSF, and GLinear. Each uses three seeds. These are manually selected examples after development, and method fidelity remains unchecked. |
| 1:23–1:38 | 7/8 development; separate 0/1 held-out | With authors' code as reference, seven of eight development units matched the reference verdict. This is in-sample. The separate RLinear held-out check matched zero of one. These counts are not a general success rate. |
| 1:38–1:50 | Real What-if slider; fixed measurements; v4 control count | Change the claim. The measurements stay. Separately, benchmark version four tested forty-eight planted false targets. None was confirmed. That benchmark used the earlier paper-only mode. |
| 1:50–2:03 | Certificate export button → actual A4 print preview | Follow the source, the generated code, and every measurement. Export the certificate through the browser's Save as PDF option: a portable record of the result, setup, and limitations. |
| 2:03–2:18 | GLinear inference receipt; warnings visible | GLinear measured zero point three seven nine six against zero point three eight two zero, for about twenty-one cents of gross inference cost. Tavily is separate. The verdict is preliminary; automated source checks retain their warnings. |
| 2:18–2:30 | Motto and repository link | Use the evidence to decide what to investigate before building on a paper. Explore the recorded checks in Nullius. Take nobody's word for it. |

## Facts to display

All three examples: ETTh1, forecast horizon 96, three seeds, tolerance ±0.01 MSE. Explain MSE as **prediction error; lower is better**. “Target reached” always accompanies **preliminary · method fidelity not checked**.

| Paper | Claimed MSE | Measured mean ± standard deviation | Gross inference cost | Evidence route |
|---|---:|---:|---:|---|
| TSMixer | 0.361 | 0.3624 ± 0.0003 | $0.2354 | `#/run/disc_2303_06053--20261009-201230` |
| SparseTSF | 0.359 | 0.3653 ± 0.0034 | $0.2035 | `#/run/disc_2405_00946--20261009-220822` |
| GLinear | 0.3820 | 0.3796 ± 0.0003 | $0.2047 | `#/run/disc_2501_01087--20261010-161916` |

Sources: exported dossiers and event logs under `web/public/data/`; [GLinear completion and limits](paper_candidates_20261010.md); [author-code evaluation](../benchmark/experiments/results/recall_cg_v1.md); [v4](../benchmark/experiments/results/benchmark_v4.md). Do not pool examples with the eight development units or with held-out RLinear. 7/8 means **agreement with the reference verdict**, including a partial reference verdict; it does not mean seven fully reproduced papers. Author-code controls: 0/9 confirmed, including one without a verdict; do not say all nine were rejected. v4 controls: 0/48 confirmed, separate configuration.

GLinear's target and table column were checked manually before the run; automatic target extraction returned no JSON, and automatic column assignment was not checked. TSMixer also retains protocol/table warnings. Reaching the number does not establish an identical author re-run. Do not hide warnings in the certificate.

Current models: Super for planning, writing, repairing and setup extraction; Lightning for target extraction. Historical Ultra runs must be dated if shown. The new main sequence uses GLinear with current roles.

## Release and human test gate

- Demo hostname suggested by the user: **https://nullius.akca.dev**, not published yet. Use the repository as the film's current closing link; replace only after the demo is reachable.
- Produce the MP4 and check complete decoding, length under three minutes, captions at small player size and the actual PDF preview. Listen to the final audio before publication; technical audio checks do not replace listening.
- One person tests the finished product next, using [product_test_20261010.md](product_test_20261010.md). Record observations before deciding further live-demo steps or author outreach. No outreach commissioned or sent.
- **No new paid paper runs.** TimeBase, then SOFTS, remain candidates. No experiment is currently running.

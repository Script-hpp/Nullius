# Benchmark history (v2, v3, v4)

Moved from the README without shortening; only relative links were adjusted. The README keeps the v4 result as the main one. Every version counts **axis 1** ("target value reached according to the harness"); **method fidelity was not checked** in any of them. Counting rule and pre-registration: [`benchmark/experiments/benchmark_v2.md`](../benchmark/experiments/benchmark_v2.md).

## Benchmark v2

Per unit (majority of three generations):

| Unit (task) | Reference verdict (source) | Majority without Tavily | Majority with Tavily | Placebo detected without / with |
|---|---|---|---|---|
| Kipf & Welling 2017, GCN (Cora) | reproduced (R) | no majority | reproduced | failure / yes |
| Wu et al. 2019, SGC (Cora) | reproduced (R) | reproduced | partially | yes / yes |
| Zeng et al. 2023, DLinear (ETTh1) | reproduced (R) | reproduced | reproduced | yes / yes |
| Zeng et al. 2023, NLinear (ETTh1) | reproduced (R) | no majority | no majority | yes / yes |
| Shchur et al. 2018, GCN, random splits (Cora) | reproduced (R, H) | partially | no majority | yes / yes |
| Shchur et al. 2018, GAT, random splits (Cora) | reproduced (R, H) | failure (setup) | no majority | failure / yes |
| Veličković et al. 2018, GAT (Cora) | reproduced (R) | not reproduced | no majority | yes / yes |
| Veličković et al. 2019, DGI (Cora) | partially (R) | not reproduced | failure (setup) | yes / failure |
| **Units agreeing with the reference verdict** | | **2 of 8** | **2 of 8** | **6 of 8 / 7 of 8** |

**Result** (all 48 runs, code `d71a8af`, [`benchmark/experiments/results/benchmark_v2.md`](../benchmark/experiments/results/benchmark_v2.md)): with Tavily, 2 of 8 paper-claim units matched the reference verdict, without Tavily 2 of 8. **Method fidelity was not checked.**

What the numbers say:
- **Agreement is low, and Tavily makes no measurable difference** at this sample size. Many misses are near misses rather than wrong numbers: the median distance to the target value over valid generations is 0.017 without and 0.011 with Tavily (post hoc, n = 20 each), against tolerances of 0.01; "partially" (within three times the tolerance) is about as frequent as "reproduced" among single verdicts (14 vs 13 of 48). In 6 of 16 unit-arm pairs the three generations disagree ("no majority"), so stability, not only the value, limits agreement.
- **Failures of the agent are common:** 5 of 24 real-claim verdicts without and 6 of 24 with Tavily are failures ("not reproducible (setup)": code never ran within the attempt limit; "agent fault suspected": result worse than a trivial baseline). Example: in one Shchur-GAT run the repair answer of the model contained no code, so the loop stopped after one attempt.
- **No placebo passed.** None of the 48 placebo verdicts (deliberately wrong target value) was "reproduced"; one was "partially". Units where the placebo majority was a failure count as not detected.
- Cost of the whole benchmark: 1.04 $ gross without, 2.56 $ with Tavily (model cost only).

## Benchmark v3

**Benchmark v3** (all 48 runs, code `4bab96e`, pre-registered in [`benchmark/experiments/benchmark_v3.md`](../benchmark/experiments/benchmark_v3.md), results in [`results/benchmark_v3.md`](../benchmark/experiments/results/benchmark_v3.md)): after a failure analysis of v2 we added fixes for code extraction, repair after timeouts, Tavily error search (role 5), repository search (role 1), following local imports and extracting implementation details. Agreement stayed at **2 of 8 units in both arms**; the units that agree changed (with Tavily, Kipf-GCN and NLinear now agree in all or most generations, SGC got worse because repository defaults replaced the tuned value). Failures of real claims with Tavily fell from 6 to 3 of 24, and single "reproduced" verdicts rose from 7 to 8. **No placebo was ever "reproduced"** (0 of 48 in v3, 0 of 48 in v2). Every miss is listed with its cause in [`results/failure_log_v3.md`](../benchmark/experiments/results/failure_log_v3.md) and [`failure_patterns_v3.md`](../benchmark/experiments/failure_patterns_v3.md). Method fidelity was not checked.

## Benchmark v4

**Benchmark v4** (all 48 runs, code `bf520c1`, pre-registered in [`benchmark/experiments/benchmark_v4.md`](../benchmark/experiments/benchmark_v4.md), results in [`results/benchmark_v4.md`](../benchmark/experiments/results/benchmark_v4.md)): after the v3 failure patterns we added three code candidates per claim, chosen only by a validation metric that our harness computes from its own validation labels (test values and the paper's target never enter the choice), checks for prediction-to-node mapping and forbidden libraries, marked repository defaults, and a comparison with the original code (with Tavily only). **Our pre-registered expectation (more agreement than v3 in at least one arm) was not met:** with Tavily 2 of 8 units agree (Kipf-GCN and Shchur-GCN), without Tavily 1 of 8 (DLinear). What changed is stability: no unit is left without a majority (v3: 4 unit-arm pairs), only 1 of 48 real-claim verdicts is a failure (v3: 9), and **all 16 placebo majorities say "not reproduced"** (v3: 14 of 16, v2: 13 of 16). Most misses are now consistent near misses ("partially"; median distance to the target 0.017 with and 0.029 without Tavily, post hoc). The validation choice picked a candidate other than the first in 54 of 96 claims. One process died on a sandbox connection error and was restarted with unchanged code ([`results/benchmark_v4_incidents.md`](../benchmark/experiments/results/benchmark_v4_incidents.md)). Cost 7.78 $ gross for 48 runs. Every miss: [`results/failure_log_v4.md`](../benchmark/experiments/results/failure_log_v4.md). Method fidelity was not checked.

## Comparison

| | v2 | v3 | v4 |
|---|---|---|---|
| Units agreeing with the reference, without / with Tavily | 2 / 2 of 8 | 2 / 2 of 8 | 1 / 2 of 8 |
| Unit-arm pairs without a majority | 6 of 16 | 4 of 16 | 0 of 16 |
| Failed real-claim verdicts (no measurement), of 48 | 11 | 9 | 1 |
| Placebo majorities "not reproduced", of 16 | 13 | 14 | 16 |
| Placebo verdicts "reproduced", of 48 | 0 | 0 | 0 |
| Cost, gross | 3.60 $ | 3.10 $ | 7.78 $ |

An earlier, frozen benchmark run (v1, four papers, product configuration off) is kept as history in [`research_log.md`](research_log.md).

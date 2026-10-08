# Tavily experiments, part 1 result (DLinear fixture rehearsal)

Pre-registration: `benchmark/experiments/tavily_experiments_test.md` (part 1). Run: `runs/zeng_dlinear_2023/20261007-224416` (started by the previous worker, finished 2026-10-07 23:51; completed, so it was not repeated). Fixture mode, Token Factory, Nebius sandbox, full mode, one run. Not a benchmark, no accuracy claim.

## Criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Run completes; `dossier.json` with `objection_checks`, `events.jsonl` with the four new event types, `zeugnis.md` with "Objections checked" | **not met** (strict) | `run_end` present; `objection_checks` 4 entries; `zeugnis.md` line 19 has the section. Events: `objection_classified` x4, `objection_checked` x4, `objection_round_end` x1; the fourth type `objection_experiment` is emitted only when an experiment runs, and 0 ran (criterion 3), so only 3 of 4 types occur. |
| 2 | Every entry keeps a verbatim quote found in the stored fixture page and its URL | **met** | all 4 quotes and URLs found in `benchmark/fixtures/role3_zeng_objections.json` (substring check) |
| 3 | At least 1 entry non-`not_measurable` and a `seed_variance`/`parameter_change` status resting on at least 2 valid own-branch measurements | **not met** | types: 1 `reported_number`, 3 `not_measurable`; 0 `seed_variance`, 0 `parameter_change`; `objection_round_end`: `n_experiments = 0`. The only measurable finding (issue #117) is a reported number, no run by rule. |
| 4 | Each status follows the fixed rule applied to the stored numbers | **met** | obj1: `same_setup = "no"` -> `offen` (rule needs `yes`); also `|ours - reported| = 0.033 > tol 0.01`, `|ours - target| = 0.084 > tol`, so no rule branch for `bestaetigt`/`widerlegt` applies. obj2-obj4: `not_measurable` -> `nicht pruefbar` |
| 5 | Main verdict from main attempts only | **met** | verdict `nicht reproduziert`, mean 0.4592, std 0.0353; recomputed from the 5 main `abweichung` attempts (seeds 0-4: 0.4665, 0.5173, 0.4314, 0.4457, 0.4350) = 0.4592. `objection_checks` entries have empty `attempts`. |

Part 1 overall: criteria 1 (event type `objection_experiment` absent because no experiment ran) and 3 not met, so part 1 is **not met** as pre-registered. Criteria 2, 4, 5 met.

## Numbers

- Cost of run: 0.2018 USD net, 0.2401 USD gross (`run_end`). Objection round: 0.0019 USD net, 8.9 s, 4 findings, 0 experiments. Tavily credits: 0 (stored fixture).
- `logs/costs.jsonl` does not exist in this worktree (`logs/` is an empty directory here, not a symlink), so costs come from events only. The cap bookkeeping for part 2 must read the main checkout's log or sum events.
- Counts per type: reported_number 1, not_measurable 3. Per status: offen 1, nicht pruefbar 3.

## Observations

- Expectation in the pre-registration was a `parameter_change` for "input window 336 vs 96". The stored findings contain no such objection (obj4 mentions "bigger window" without a value, correctly `not_measurable`). Stored fixture offers no measurable seed/parameter finding for DLinear, consistent with `results/role3_dlinear_stage1.md`.
- Issue #117: model classified `same_setup = "no"`, not `unknown` as expected. Not an error under the pre-registration (only `yes` would be). Status `offen` either way.
- obj3 `type_reason` is empty and note "classification missing or invalid": classifier output for that finding was invalid and fell back to `not_measurable`.
- The main run itself did not reproduce the target (0.459 vs 0.375, 9 of 13 attempts were `fehler` with repair, 4 seed attempts `abweichung`). Within scope of this test; not a feature problem. It did not re-run.
- Conclusion: pipeline runs end to end and demotes correctly, but part 1 cannot show a measured objection experiment on DLinear. Seed/parameter path remains untested on real data until part 2.

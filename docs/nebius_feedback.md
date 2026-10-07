# Feedback on Nebius Sandboxes (contree-sdk 0.3.6) from building the paper reproduction agent

Collected during the hackathon; every item was observed in this repository, with the file or commit where it shows up. Written for the Nebius team. Devpost lists "feedback on Nebius Token Factory, AI Cloud and any NVIDIA tools, models" as a required submission component (submission deadline 30 Oct 2026, 10:00 PDT); this file is the draft for it.

## 1. Status polling of long operations hits the 10 s HTTP timeout under parallel load (reproducible)
- **What:** `session.run(...).wait()` polls `GET /sandboxes/v1/operations/<id>`. `ContreeConfig.transport_timeout` defaults to 10 s. With about 8 concurrent operations (two processes with 3 to 4 parallel sandbox branches each, benchmark runs of 20 to 60 s per operation) single poll requests did not answer within 10 s, and the whole `wait()` failed with
  `contree_sdk.sdk.exceptions.api.ApiTimeoutError(request=RequestInfo(url='https://api.tokenfactory.nebius.com/sandboxes/v1/operations/<id>', method='GET'), response=None, timeout_type='read')`.
  It ended three benchmark processes within two attempts (also once with the laptop awake), each after minutes of otherwise healthy runs.
- **Detail from the fourth abort:** the error object then carried a `ResponseInfo` with headers (`content-type: application/json`, `transfer-encoding: chunked`, `content-encoding: gzip`): the server had started answering, and the body stalled for more than the 10 s read timeout. The three first aborts had `response=None`.
- **Consequence in the SDK:** after the error the session object is unusable. Reusing it raises `ContreeImageStateError: Image state "None" cannot have state FAILED` (verified against the real API by injecting one timeout into `get_operation_status`). The operation itself keeps running on the server, but the client cannot pick it up again because the operation id is not exposed by `session.run(...)`.
- **Our workaround:** `transport_timeout=60`, and on the error we repeat the whole attempt on a fresh branch (`src/repro_agent/sandbox/nebius.py`, `loop.py::_attempt`).
- **Suggestions:** retry a failed status poll inside `_wait_operation` (polling is idempotent), or raise a distinct error that keeps the operation id so the caller can resume; a larger default read timeout for status polls; document the recommended concurrency.

## 2. `session.read()` returns the base image, not the branch state
`session.read()` reads the image the session started from; the session uuid is never updated after `run()`. We had to read result files by gzip + base64 through a `run` call, chunked, because stdout is capped (`nebius.py::get_file`, commit d98a7cc). A `read()` that follows the session state would remove about 40 lines of workaround.

## 3. Chained runs need `disposable=False`
Chained `session.run()` calls fail unless `disposable=False` is passed explicitly (commit d98a7cc). Making this the default for sessions, or a clearer error, would help.

## 4. Timeouts come back as data, not as an exception
A server-side run timeout does not raise; it arrives as `state.timed_out` with `exit_code -1` and signal 9. A client-side operation timeout raises `OperationTimedOutError`. Two different signals for "took too long"; we handle both (`nebius.py::run`).

## 5. Output limits
The SDK default `default_truncate_output_at` is 64 KiB and silently truncates stdout; the server maximum is just under 10 MiB (`_MAX_OUTPUT` in `nebius.py`). A documented constant or a clear warning on truncation would avoid silent data loss.

## 6. No network switch for a sandbox run
There is no SDK option to run a command without network. We isolate untrusted agent code with `unshare -n` inside the command, and `resolv.conf` set in a snapshot does not persist. A per-run `network=False` would make the isolation cleaner and verifiable.

## 6b. Long benchmark runs: what the workaround from item 1 looks like in practice
Benchmarks v2 and v3 (2026-10-01) each ran 48 agent runs in three parallel processes over several hours, with up to 9 concurrent sandbox operations. Even with `transport_timeout=60`, v3 logged 6 `ApiTimeoutError` on status polls (`GET /sandboxes/v1/operations/<id>`, `timeout_type='read'`, `response=None`), 5 during `run` and 1 during `put_files` (`logs/bench_v3/*.log`, local). All 6 were recovered by repeating the attempt on a fresh branch; no run was lost. Without the workaround each of them would have ended a process. Item 1 is therefore not a one-off but a steady rate under real load (roughly one per 8 runs).

## 7. What worked well
- Real branching from one snapshot is exactly the abstraction the agent needed; 20 parallel branches from one snapshot ran without errors in our test.
- `apply_files` with several files in one call keeps the operation count low.
- Snapshots by tag make cleanup scripts easy (`scripts/cleanup_snapshots.py`).

## Token Factory and NVIDIA models (observed)
- **Nemotron 3.5 Lightning spends the whole token budget on reasoning for a simple extraction.** A table-extraction prompt with `max_tokens=6000` returned `finish_reason: "length"` with `reasoning_tokens: 6000` and no JSON (logged in `logs/costs.jsonl`); the model writes its thinking into the answer. We now use more tokens, split inputs into small sections and use Super for extraction. A switch to turn reasoning off or cap it would help.
- **Usage accounting matched the console.** Token counts from the `usage` field of the API responses agreed with the console usage page within its rounding (0.01 M tokens); net 1.256 $ in our log vs 1.26 $ in the console (`scripts/cost_report.py`, `benchmark/experiments/`). The response has no cost field, so we compute cost from the list prices of `GET /v1/models`; a `cost` field (as OpenRouter returns) would remove that step.
- **Console lag at the start:** on the evening of 28 Sep the usage page showed "No usage" although Nemotron calls had already run; the costs appeared correctly the next morning (05:35 UTC on 29 Sep). The exact delay is unknown, but it was several hours, which made it impossible to verify spend before the next day.
- **Trial end blocked the promo credits (confusing for new users):** the console showed "Billing: Suspended" with the note that the trial had ended and usage was stopped according to the after-trial setting, plus a "Switch to paid usage" button. The account still held 29.75 $ of promo credit, but all calls were blocked because only the 1 $ trial credit was used up; the credit became usable only after switching to "paid usage". The interface does not say that "paid usage" here just means using the existing credit and does not charge the card immediately. We do not have the exact API response for the blocked calls; a distinct, documented status code and message for billing problems would let clients stop instead of retrying (our client treats 401/402/403 and billing keywords as fatal).
- **VAT is added on top of the list prices** (19 % in our case: net 1.26 $, VAT 0.24 $, gross 1.49 $), which is easy to overlook when planning a credit budget; the price list and the API give net prices only.
- **Models used:** `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b` via the OpenAI-compatible endpoint; ID casing differs between models (`Nemotron-3_5-Lightning` vs `nemotron-3-super-...`), which cost us a lookup.

## Reasoning tokens and truncated answers (measured from our local cost log)
Evaluated from `logs/costs.jsonl` (local, not in Git) with a throwaway script outside the repository. Filter, exactly: rows with `provider == "tokenfactory"`, `cache_hit` false, a `reasoning_tokens` field, and `ts` before 2026-09-30 23:00:00 UTC. That gives 1,094 calls from 2026-09-29 07:17 UTC to the last matching row at 2026-09-30 22:54 UTC (earlier rows lack the field; later rows, including the running benchmark, are excluded). An earlier version of this section had no fixed end time and counted 22 more Super calls that were written to the log while it was being evaluated. `completion_tokens` from the API already contains the reasoning tokens, so they are billed as output.

| Model | Calls | Completion tokens | Reasoning tokens | Share | Calls with `finish_reason: length` |
|---|---|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | 996 | 1,395,759 | 869,958 | 62.3 % | 4 (0.4 %) |
| `nvidia/Nemotron-3_5-Lightning` | 11 | 40,587 | 40,109 | 98.8 % | 3 (27 %) |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | 87 | 138,886 | 94,227 | 67.8 % | 0 |

- **Most output tokens are reasoning.** Every call in the sample reported `reasoning_tokens` greater than zero. At the list prices we recorded on 2026-09-28 (output per million tokens: Lightning 0.24 $, Super 0.90 $, Ultra 3 $), the reasoning tokens correspond to about 0.78 $ (Super), 0.28 $ (Ultra) and 0.01 $ (Lightning) of net output cost in this sample. A documented per-request switch or cap for reasoning (for example a budget parameter) would make this cost controllable; today we can only raise or lower `max_tokens`.
- **Lightning:** in 3 of 11 calls the whole budget (6,000 tokens) went to reasoning and the answer was empty (`completion_tokens` 6000, `reasoning_tokens` 6000; 2026-09-29 08:10, 08:11 and 2026-09-30 18:43 UTC). The sample is small; the 27 % is not a rate estimate. This is the case already described above.
- **Super:** 4 calls hit `length` at `max_tokens` 8192 with 2,767 to 6,044 reasoning tokens (all on 2026-09-30 between 22:00 and 22:44 UTC, while the agent wrote experiment code). The answer was cut off inside the code. We now retry such calls with a larger limit and never cache truncated answers (commit `530f748`).
- **Limit of this evaluation:** one project, one workload (code writing and table extraction), about two days; the shares depend on our prompts and may differ elsewhere.

## Running cost of a full benchmark (measured)
A benchmark of 48 agent runs (8 paper claims x 2 arms x 3 generations, each with a placebo claim, 5 to 30 seeds each) cost 3.60 $ gross (v2) and 3.10 $ gross (v3) on Token Factory, about 0.07 $ per run (`runs/benchmark_v2.jsonl`, `runs/benchmark_v3.jsonl`, field `cost_usd_run`). Over the whole project the local log counts 28.38 $ gross on Token Factory by 2 Oct 2026 (`docs/budget_plan.md`, "Stand"), including later benchmark runs, trials and development. Low prices for Super made it affordable to run every agent change through the full benchmark instead of guessing; this is where Token Factory sped up our workflow most.

## Nemotron Super as a code writer (observed in the benchmarks)
- **Missing code block:** in v2, 7 of 96 verdicts (48 runs) ended because a repair answer contained reasoning but no code block (the model described the fix instead of returning the script). We now accept other fences and unfenced code that parses, and rewrite from the plan if the answer still has no code (commit `f6f4222`).
- **Instructions about unavailable libraries are ignored:** in v2 and v3 the model imported `torch_geometric` in all four attempts of a run, also after an explicit repair hint that it is not installed (`benchmark/experiments/failure_patterns_v3.md`, M3). Placing the hint in the first prompt is the next mitigation (v4).
- **Index bugs that survive repair:** the model mapped predictions to node ids by position instead of by index (`zip(test_ids, pred)` instead of `pred[test_ids]`), giving chance-level accuracy; three repair rounds with the error description did not fix it (M1). A deterministic check on validation data now catches it before the full run.


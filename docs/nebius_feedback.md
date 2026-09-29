# Feedback on Nebius Sandboxes (contree-sdk 0.3.6) from building the paper reproduction agent

Collected during the hackathon; every item was observed in this repository, with the file or commit where it shows up. Written for the Nebius team. Devpost lists "feedback on Nebius Token Factory, AI Cloud and any NVIDIA tools, models" as a required submission component (submission deadline 30 Oct 2026, 10:00 PDT); this file is the draft for it.

## 1. Status polling of long operations hits the 10 s HTTP timeout under parallel load (reproducible)
- **What:** `session.run(...).wait()` polls `GET /sandboxes/v1/operations/<id>`. `ContreeConfig.transport_timeout` defaults to 10 s. With about 8 concurrent operations (two processes with 3 to 4 parallel sandbox branches each, benchmark runs of 20 to 60 s per operation) single poll requests did not answer within 10 s, and the whole `wait()` failed with
  `contree_sdk.sdk.exceptions.api.ApiTimeoutError(request=RequestInfo(url='https://api.tokenfactory.nebius.com/sandboxes/v1/operations/<id>', method='GET'), response=None, timeout_type='read')`.
  It ended three benchmark processes within two attempts (also once with the laptop awake), each after minutes of otherwise healthy runs.
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

## 7. What worked well
- Real branching from one snapshot is exactly the abstraction the agent needed; 20 parallel branches from one snapshot ran without errors in our test.
- `apply_files` with several files in one call keeps the operation count low.
- Snapshots by tag make cleanup scripts easy (`scripts/cleanup_snapshots.py`).

## Token Factory and NVIDIA models (observed)
- **Nemotron 3.5 Lightning spends the whole token budget on reasoning for a simple extraction.** A table-extraction prompt with `max_tokens=6000` returned `finish_reason: "length"` with `reasoning_tokens: 6000` and no JSON (logged in `logs/costs.jsonl`); the model writes its thinking into the answer. We now use more tokens, split inputs into small sections and use Super for extraction. A switch to turn reasoning off or cap it would help.
- **Usage accounting matched the console.** Token counts from the `usage` field of the API responses agreed with the console usage page within its rounding (0.01 M tokens); net 1.256 $ in our log vs 1.26 $ in the console (`scripts/cost_report.py`, `benchmark/experiments/`). The response has no cost field, so we compute cost from the list prices of `GET /v1/models`; a `cost` field (as OpenRouter returns) would remove that step.
- **Console lag at the start:** the usage page showed nothing for the first hours of use, which made it impossible to verify spend before the next morning.
- **Trial end:** the account showed "Billing: Suspended" after the trial ended. We do not have the exact API response; a distinct, documented status code and message for billing problems would let clients stop instead of retrying (our client treats 401/402/403 and billing keywords as fatal).
- **Models used:** `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b` via the OpenAI-compatible endpoint; ID casing differs between models (`Nemotron-3_5-Lightning` vs `nemotron-3-super-...`), which cost us a lookup.

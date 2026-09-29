# Feedback on Nebius Sandboxes (contree-sdk 0.3.6) from building the paper reproduction agent

Collected during the hackathon; every item was observed in this repository, with the file or commit where it shows up. Written for the Nebius team.

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

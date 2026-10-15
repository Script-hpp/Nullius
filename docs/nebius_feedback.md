# Feedback on Nebius Token Factory, Sandboxes and NVIDIA Nemotron

From building Nullius, an agent that rebuilds ML papers in Nebius Sandboxes and checks the numbers with an independent harness. Every item below has an evidence ID (E1, E2, ...) that points to a file, line, commit or log in this repository; see [`docs/audits/nebius_feedback_evidence_20261015.md`](audits/nebius_feedback_evidence_20261015.md) for the exact lines and how each was checked. Items we could not prove are listed separately at the end and are not part of the feedback proper.

Context: contree-sdk 0.3.6 (E1), models `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b` and, until 2026-10-09, `nvidia/Nemotron-3-Ultra-550b-a55b` via the OpenAI-compatible endpoint. All findings are from our workload (code writing, table extraction, CPU training in sandboxes), 2026-09-28 to 2026-10-15.

## Top 5 by impact

| # | Finding | Cost to us | Suggestion | Evidence |
|---|---|---|---|---|
| 1 | **Model removed without a signal we could see.** Ultra answered `chat/completions` with a plain HTTP 404 from about 01:52Z on 2026-10-09; `GET /v1/models` no longer listed it. | A running benchmark aborted: two runs died, five workers were stopped, 1.96 $ gross spent to the halt, about 1.4 $ of it lost. Roles had to be re-pointed to Super, and the run was restarted. | Announce removals ahead of time; keep the old ID answering for a transition period; return a structured deprecation error (410 plus successor ID) instead of 404; offer a stable alias. | E2, E3 |
| 2 | **Sandbox stdout/stderr is cut at 65535 bytes, keeping the head, and nothing in the result makes this obvious** (`default_truncate_output_at = 65535`; the `truncated` flag exists but is easy to miss). | A 1410-path file listing (83540 bytes) lost the three files that decide the experiment setup; the live run then used an incomplete setup (0 of 3 seeds measured, no verdict). Tracebacks of long training logs sit at the end and were cut. | Raise the default or fail loudly; return the tail by default for stderr (or both head and tail); document the constant next to `run()`. | E4, E5 |
| 3 | **A single status-poll read timeout (10 s default) kills the operation and the session.** `session.run(...).wait()` raised `ApiTimeoutError` under about 8 parallel operations; the session is then `FAILED` and the operation id is not exposed, so it cannot be resumed. | Four benchmark process aborts on 2026-09-29 (two processes, each restarted once); since then every attempt needs a retry on a fresh branch. | Retry idempotent status polls inside the SDK; keep the operation id in the error so callers can resume; document recommended concurrency. | E6, E7 |
| 4 | **Reasoning tokens dominate output cost, and we found no way to cap them.** In two local logs (2026-10-09 to 2026-10-15) 57 to 74 % of Super's completion tokens and 99 % of Lightning's were reasoning; 18 of 28 Lightning calls ended with `finish_reason: length`. | We raise `max_tokens` (up to 24000) and retry; the answer text of Lightning often holds its thinking. | A per-request reasoning switch or budget; return reasoning separately from the answer content. | E8, E9, E10 |
| 5 | **No network switch per run.** We found no SDK option to run a command without network. | We wrap every untrusted run in `unshare -n` ourselves, and cannot verify the isolation from the platform side. | `network=False` (or an allow-list) per `run()` that the result reports back. | E11 |

## Token Factory

- **Billing state was unclear at trial end.** After the trial the console showed `Billing: Suspended` and calls were refused. The exact API response was not saved, so our client treats HTTP 401/402/403 and billing keywords as fatal instead of retrying (E12). Suggestion: a documented, distinct status code and error code for billing problems, so clients can stop cleanly.
- **Console and client accounting agreed.** Our log (token counts from the `usage` field times list price) matched the console within rounding on 2026-10-02: 30,25 $ vs 30,24 $ gross, Super input/output tokens identical (E13). Please keep this property. The response carries no cost field, so every client has to multiply by the price list; a `cost` field would remove that step.
- **Net prices, VAT on top.** In our account the console showed net 25,43 $, VAT 4,83 $, gross 30,25 $ (E13). The price list and the API are net; budget limits have to be gross. A gross figure or VAT hint on the usage page would help budgeting.
- **Truncated answers without `finish_reason: length`.** In one live run six code answers broke off mid-script, with call costs in the range of complete answers and no `length` finish reason (E14). We could not establish the cause (the cap, sampling temperature 0.8 to 1.4 and the model are all candidates), so treat this as a lead, not a verified API defect.
- **Model ID casing differs between models** (`Nemotron-3_5-Lightning` vs `nemotron-3-super-120b-a12b`), which cost us a lookup against `GET /v1/models` (E15). Case-insensitive matching would remove this.
- **Vision/PDF:** Nano Omni is only available as a dedicated model, so we had to restrict ourselves to HTML/LaTeX sources and no figure analysis (E16).

## Nebius Sandboxes (contree-sdk 0.3.6)

- **`session.read()` returns the base image, not the branch state.** We read result files through `run` with gzip, a SHA-256 and base64 in chunks (E17). A `read()` that follows the session state would remove this code.
- **Chained runs need `disposable=False`** (E17). A default or a clear error message would help.
- **Timeouts arrive in two shapes:** a server-side run timeout comes back as data (`state.timed_out`, exit code -1), a client-side one raises `OperationTimedOutError` (E18).
- **Capability question for isolation.** Our forecast task needs to drop privileges inside the sandbox with `setpriv` and `chown`. This is verified in our task image but not in the Nebius backend; it stays an open preflight (E19). Documenting which capabilities and tools a sandbox guarantees would let us decide without a paid probe.
- **CPU-only sandbox for us.** A claim (SST) could not run because the authors' model imports `mamba_ssm`, a CUDA kernel library. Every generated candidate either failed the import or rewrote the scan by hand and crashed on tensor shapes. We now block such claims before paid steps (E20). If a GPU sandbox flavor exists or is planned, an opt-in path would extend what a paper-checking agent can reproduce.

## NVIDIA Nemotron

- **Super (code writing).** Large share of tokens goes to reasoning (E8). In our benchmark failure analysis, 7 of 23 failures in v2 were repair answers with reasoning but no code block (E21); the model ignored an explicit "library not installed" hint in all four attempts of one run (E22). Both are mitigated in our loop; we report them as observations, not as a comparison with other models.
- **Super (structured output).** A single backslash in a JSON string (`\textbf`, `\bst`) decodes to a control character (`\b`, `\t`), so quotes copied from LaTeX no longer matched the source; three papers were rejected in one live discovery before we repaired the escapes (E23). We did not test whether other models do better.
- **Temperature.** Our candidate schedule raised the temperature to 0.8 to 1.4 for later candidates; at 1.1 a SyntaxError in an f-string and at 1.4 multilingual word salad appeared. Capping at 0.7 fixed it (E14). Recommended range for code generation in the model card would have saved us this run.
- **Lightning.** Used for extraction and triage. It spends its token budget on reasoning (E9, E10); in one table extraction three attempts (`max_tokens` 6000, 12000) returned no valid JSON, two of them ending on `length` (E24). Our interpretation: usable for short triage, not for extraction.

## Docs and developer experience

- Please document, next to the branching guide, the constants that bit us: `transport_timeout` default 10 s (E6) and the 65535-byte output cut (E4).
- Please document the `session` vs `image` semantics: image `.run()` returns a new image, session `.run()` mutates the session. We had to establish this by experiment and wrote it into our adapter docstring (E1).

## What worked well

- **Real branching** from one snapshot is exactly the primitive an experiment agent needs. Twenty parallel branches ran without error in our rate-limit check on 2026-09-28 (E25).
- **`apply_files`** with several files in one call keeps operation counts low (E26).
- **Price made benchmarking affordable.** A 48-run benchmark cost 3,60 $ and 3,10 $ gross (v2, v3), so every agent change could be run against the full set (E27).
- **Usage numbers were reliable** (E13).

## Unverified / anecdotal (not part of the feedback)

These came from our own observations during the hackathon but we cannot back them with a file in the repository. Treat as hints.

- Console showed "No usage" on the evening of 2026-09-28 although calls had run; the numbers appeared the next morning. No screenshot or log was kept.
- The "Billing: Suspended" console text and the "Switch to paid usage" path that released promo credit are known to us only from the account owner's report; the repository records the text but not the exact UI wording or the API response (E12).
- That no deprecation notice reached us before 2026-10-09: absence of evidence, we have no complete inbox/console record.
- Roughly one `ApiTimeoutError` per 8 runs in benchmark v3 (6 events): the per-run logs referenced for this count are no longer available locally; only the four 2026-09-29 aborts are documented (E6).
- About 91 % uptime and 36 tokens/s on the free OpenRouter endpoints (CLAUDE.md, prototype phase): no measurement is stored in the repository.
- Token Factory charges the card automatically when the balance goes negative: we plan with this (E28) but never triggered it.

## Devpost text field (about 150 words)

> **Token Factory / Sandboxes / Nemotron feedback.** Branching in Nebius Sandboxes and low-priced Nemotron Super made it affordable to run a 48-run benchmark per agent change (3.10 to 3.60 $ gross). Console and client cost accounting matched within rounding. What hurt: (1) Nemotron Ultra disappeared with a bare HTTP 404 on 2026-10-09 and aborted a running benchmark (about 1.4 $ lost); please announce removals, return 410 plus successor, or offer an alias. (2) Sandbox stdout/stderr is cut at 65535 bytes by default, keeping the head; a 1410-file listing lost the files that mattered. (3) One 10 s status-poll timeout kills the session and hides the operation id. (4) Reasoning tokens were 57 to 99 % of output in our logs, and we found no way to cap them. (5) We found no per-run network switch. Every point links to a file or commit in our repo (`docs/nebius_feedback.md`, evidence table in `docs/audits/`).

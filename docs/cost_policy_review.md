# Cost policy review — 10 Oct 2026

Read-only review of accounting and admission controls; no paid API calls, no dashboard access, no change to production limits. References are the code in this checkout and the documented configuration, not a verified current balance.

## Follow-up: local CycleNet trial, 10 Oct

The user supplied a console screenshot showing $13.50 account balance, last updated 12:14 UTC. The usage range is 10 September–10 October, not one day's spending. The earlier allocation was $12 for the public demo and approximately $5 for local work. Conservatively preserving the full $12 demo reserve leaves $1.50 for new local work at that snapshot. A promised $10 top-up after a successful reproduction is conditional and is not available budget.

The shared main-worktree `logs` directory was a symlink to itself; dependent worktrees therefore could not read the accounting history. Restored its real directory from `assistant/logs_local_backup`, preserving all 721 cost records and caches byte-for-byte. Both the original backup and the broken link (renamed `logs_broken_link_20261010`) remain available. The recovered cost history ends at 9 October 20:20:39 UTC and accounts for $1.360913 gross since the configured 11:11 UTC cutoff. It does **not** prove that later spending was zero. The existing $5 local baseline was not reset or enlarged.

Authorized exactly one manually seeded CycleNet/Linear check, three seeds, v1 Table 7 target MSE 0.378. Its temporary driver adds a serialized admission check around every actual model request, including retries: known model prices, byte-based input allowance, explicit output ceiling, and $0.50 gross inference planning limit. Failed or uncertain requests retain their reservation. Tavily SDK attempts have a separate limit of 12 estimated credits. These are trial-specific controls, not a repair of the public API's budget policy or proof of provider billing accuracy; sandbox costs are outside the inference limit. Trial evidence and its driver are retained with the run. No other shortlist candidate is authorized by this trial.

Completed trial `disc_2409_18479/20261010-154805`: **not reproduced**, mean MSE 0.5455043472, standard deviation 0.0045092745, three seeds, target 0.378, tolerance 0.01. Inference cost $0.2027830200 net / **$0.2413117938 gross**; guard accounting agrees with the client. Tavily admission counted 12 estimated credits. No further paid trial was started. Remaining local planning room from the supplied snapshot is approximately $1.2587 after inference, preserving the $12 demo reserve; later server spending or other charges would reduce it. The conditional top-up is still excluded.

Offline guard checks passed for actual-usage booking, oversized-request rejection, unknown-price rejection, timeout reservation and retry admission. The recovered accounting file retains its original 721-record byte prefix and receives new entries normally. Removed the tracked absolute `logs` symlink from Git and ignored the runtime path so future checkouts cannot recreate this self-reference; existing runtime files remain on disk. Trial controls remain local to the driver; the production gaps listed below remain open.

## What works

- Costs are recorded net; limits use gross cost including the configured VAT. Cached answers cost zero; paid empty answers are booked too.
- The LLM client warns at 80% and checks the global budget before a fresh logical call, stopping when logged spending is already at 95%.
- Public live admission checks budget and Tavily floors, daily spending, job counts and per-IP limits. Job starts are persisted and restored; logs need a persistent volume.
- Public runs force three seeds unless full runs are explicitly enabled. Replay remains available when live is paused.
- Tavily attempts, including failed retries, are conservatively counted. The documented dashboard policy keeps pay-as-you-go off.
- Separate server and development allocations avoid treating two independent logs as one shared wallet.

## Material limits

1. **The per-run dollar limit does not cover every model call.** `loop._ask` checks `MAX_COST_PER_RUN_USD`, but target extraction, setup extraction, repository file selection and community/objection extraction call `LLMClient.complete` directly. Their costs are counted, but that client enforces only the global budget. Discovery also has no per-job dollar limit. A run can spend past its cap before another `_ask` notices.
2. **Admission thresholds are not reservations.** Global checks use completed, logged calls; there is no reservation for the next request. The next bill can cross 95%. `_call_with_retry` can book paid empty responses and retry without rechecking the global budget. The run check is outside the orchestrator lock, so queued callers can pass it before earlier callers spend. Unknown Token Factory models are logged at zero estimated cost when the API supplies no cost; request token ceilings are not mandatory.
3. **The watchdog releases the slot without stopping the worker.** `api._arm_watchdog` reports an error after 45 minutes (reproduction) or 15 minutes (discovery), then releases the semaphore. The original thread continues and may spend while a new job starts. Therefore “one job at a time” is not a spending guarantee after a timeout.
4. **Tavily floors are checked only at job admission.** `CountedClient` records each attempt but has no in-job credit reservation or stop at the configured floor. A running job can consume part of the reserve. Keeping dashboard pay-as-you-go off limits monetary exposure, not exhaustion of credits.
5. **The budget documents disagree.** `docs/budget_plan.md` still includes a $20 server allocation and $0.75 run cap; the newer deployment plan uses $12 and $0.50. The recorded $17.02 console balance is from 9 Oct 11:11 UTC, before the later catalog checks; it is not today's verified balance. Five Discovery queries are now allowed per dataset while two tests and the old credit estimate assume a smaller search budget.

The deployment calculation of an absolute $11.40 server maximum is therefore a planning threshold, **not a proven upper bound**. Provider-side accounting is authoritative. Local and server spending must be reconciled before assigning new budget baselines; changing a baseline alone does not add funds.

## Priorities before deployment

1. Verify the current console balance and Tavily credits; allocate server and development shares whose sum fits that balance. Use `docs/deploy.md` / `deploy/.env.example` as the current configuration reference, with persistent logs and exactly one API process. Keep the existing open-demo/replay policy.
2. Enforce per-run accounting at the shared LLM boundary, including extraction, reporting and retries; reserve a conservative cost before each paid request and release/book it afterward. Reject unknown prices when the provider cannot supply an authoritative cost. Check limits under the same lock as admission.
3. Cancel timed-out jobs cooperatively before allowing further paid work, or keep their slot held until the worker exits. Apply a credit check before each Tavily attempt.
4. Reconcile the five-query Discovery policy with its tests and measured credit estimates. Prefer counts and invoices from existing runs; do not spend credits merely to test controls.

Items 2–4 are findings for follow-up, not changes implemented by this review. Test results do not establish production billing accuracy or current available funds.

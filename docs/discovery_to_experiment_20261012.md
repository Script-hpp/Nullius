# Discovery → Experiment, 12 October 2026

## Problem and resulting flow

Live Discovery previously streamed candidates without making them executable in the API. Reproduction accepted only entries in the curated `benchmark/catalog.yaml`. The catalog offered Start new check for unchecked entries without checking their supported task first; SimMLP was rejected only after clicking. Replay always loaded the bundled September 30 example, despite a newer live search.

The jury can now open Discover, select a saved compatible claim, press **Check this claim**, watch its new three-seed experiment, and open its certificate. Completed live candidates appear in the catalog with their actual outcome and evidence. Search alone remains separate from reproduction.

## Implementation and provenance

- `logs/live_candidates.json` stores server-owned candidate entries and outcomes using atomic writes in the existing logs volume. IDs include the paper plus a claim fingerprint: separate methods, datasets, targets and protocols do not overwrite one another. Cached discovery cannot erase an experiment outcome.
- Candidates are saved before their live event is published. The browser submits only the ID to the existing `/api/reproduce` route. The server resolves the stored claim and retains the same gates, spending limits, shared job slot, IP limits and three-seed default. Browser-supplied target or task overrides are ignored.
- `GET /api/discover/candidates` restores the latest 30 saved claims when opening Discover. Loading this list spends no credits. On the first upgrade, accepted v2/v3 entries in the existing `discovery_cache.json` are imported without model or search calls. The old cache and both benchmark YAML files remain untouched.
- `GET /api/reproduce/eligibility` uses the existing task adapter to decide whether a claim can run. Unsupported claims show a reason rather than a start button, in both Discover and Catalog. Unverified columns, protocols and input lengths retain their existing warnings and provisional interpretation.
- The live job carries its paper metadata from the queued event, so a new paper has its correct title before it has a published catalog row. Completion refreshes the catalog and index; generated artifacts, measurements and verdicts remain accessible.
- `logs/discovery_replay.json` stores the latest completed search, including start, API events, candidates, errors and completion. `GET /api/discover/replay` serves it without spending. Replay prefers this server recording and labels its date. Without one, the UI explicitly identifies the bundled demo recording; it no longer calls that example the user's last search. Replay storage failure produces a warning and still releases the job slot.

The stored SimMLP 81.4% claim cannot simply be assigned to Planetoid or Shchur-LCC. The paper's Table 4 and inductive setup report a different evaluation involving transductive and unseen nodes. Its quoted target was not changed or relabeled to make a run possible. Source: [Training MLPs on Graphs without Supervision](https://arxiv.org/html/2412.03864).

## Verification

- **147 passed:** `uv run pytest tests/test_live_catalog.py tests/test_api.py tests/test_api_repro.py tests/test_catalog.py tests/test_catalog_reference.py tests/test_export_site_data.py tests/test_web_guard.py tests/test_spend_guard.py -q`.
- **102 passed, 2 deselected:** affected Discovery tests. The two pre-existing search-limit tests documented in `discovery_failure_20261012.md` still assume three queries where unchanged production behavior permits five.
- **137 frontend tests passed**, production build passed, lint had no errors (existing warnings).
- Production-build browser checks at 1440×1000 and 390×844: saved claims, reload, live discovery, latest server replay, unsupported Cora setup, busy job, daily limit, exact claim POST, live title, certificate link, and new result visible in Catalog without reloading. Twelve scenarios, no page errors.
- Four further browser scenarios confirm September demo replay and historical negative evidence remain separately accessible in recorded and mocked live presentation.
- All experiment and search starts in verification were simulated. No paid model, Tavily or sandbox requests were made.

## Deployment and limits

Push alone does not deploy. Redeploy the application and retain the existing `/app/logs`, `/app/runs` and `/data` volumes. No new environment variables are needed. Do not clear start, cost or reservation logs to unlock tests; existing daily limits remain authoritative.

Searches before this change have no full API replay unless separately recorded. Their validated cached claims can be recovered, but their full timeline cannot be reconstructed from those claims. The next completed search creates the first current server replay. Model fidelity and success on a newly discovered paper remain unproven until its actual experiment and evidence are examined.

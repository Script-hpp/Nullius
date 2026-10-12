# SARAF diagnosis and focused safeguards, 12 October 2026

## Evidence

Run: `disc_2606_04135_e3e2931687c224d2--20261012-210402`, [public evidence](https://nullius.akca.dev/#/run/disc_2606_04135_e3e2931687c224d2--20261012-210402).

The selected code has SHA-256 prefix `31020feb2c94`. Its measured seeds are 0.5344979492, 0.5201671487 and 0.5446536697; mean 0.5331062559, sample standard deviation 0.0123024402. The run took 661.6 seconds and reported $0.2426 gross model cost. These historical values and artifacts were not changed.

Two confirmed product errors invalidate a strong conclusion about the paper:

1. Discovery selected SARAF with target 0.366, but the [paper's Table 4](https://arxiv.org/html/2606.04135) assigns 0.371 to SARAF and 0.366 to RAFT on ETTh1, horizon 96, input length 720. The subsequent extraction returned 0.371; the old catalog policy kept 0.366 and continued with a warning. The LaTeX table's multicolumn headers were not parsed.
2. The generated retrieval class has empty preparation and returns zeros. The model's retrieval branch also uses zeros, and normalization differs from the [author implementation at commit 7b49aa8](https://github.com/ShiqiaoZhou/SARAF/blob/7b49aa8aabb38e8fe63adbc3146b1890958bf3ca/models/SARAF.py). The selected candidate already contained retrieval placeholders before its runtime-error repairs. Five passing parameter checks did not establish method fidelity.

The repository comparison was restricted to selected aspects and omitted `layers/Retrieval.py` after using its character budget on earlier files. One unverified finding was dropped, yet the result was `no_deviation`. The report also incorrectly mentioned a validation fallback despite selection using `spec (5) + val`.

## Changes

- The deterministic column checker reads the enclosing LaTeX tabular, expands multicolumn spans, matches method plus MSE/MAE subcolumn, and extends a short quote to its actual source row. It checks dataset and a stated forecasting horizon, and rejects ambiguous layouts. Nested font formatting preserves column spans. SARAF 0.366 now fails; SARAF 0.371 and RAFT 0.366 pass the correct column checks.
- Catalog claims require verified target evidence. An extraction failure, a differing target or an unverified column stops before planning or experiments. No target is silently rewritten. These stops leave the claim unchecked and produce a reason, not a negative paper verdict. Benchmark input files and archived measurements remain unchanged.
- Live Discovery cache version is now 4: pre-fix cache results are read afresh on the next search. Old versions remain importable for provenance, but are subject to the strict reproduction target check. The extraction prompt explicitly prioritizes method identity over bold/highlighted baseline numbers.
- `method_guard.py` detects known retrieval placeholders against complete, non-trivial reference functions: empty bodies or functions always returning zero tensors. It runs before every seed attempt, including repaired code. Ordinary zero initialization followed by data-dependent computation remains allowed. Known placeholders go back through the bounded repair loop; exhaustion stops without a paper verdict or launching the remaining seeds.
- The author-code comparison prioritizes `models/` and `layers/` within its existing 60,000-character budget and includes core architecture and placeholders among its reviewed aspects. Truncation or discarded findings now yield `incomplete`, with explicit unchecked method fidelity. Historical `no_deviation` events carrying cuts or discarded findings are also described as incomplete by the UI; stored events are preserved.
- Live checks that stop without a verdict show **Stopped** and the reason. Catalog and Discovery retain that reason after navigation. Code rejected before sandbox execution is described accordingly. The misleading validation-fallback wording is corrected.
- An intermittent completion-order race surfaced in the final API checks: Discovery's done event could be read before its worker-slot release. Both now happen in one event-loop callback, with release first. Watchdog completion still holds the slot until the worker finishes.

## Verification and limits

Tests use mocked models and sandboxes. Saved public SARAF code, the actual LaTeX source and author files were also checked locally without executing them: the old target fails, the correct target passes, and the generated code is blocked even against the truncated repository excerpt from the original run.

Regression tests cover column swaps, nested headers, dataset/horizon ambiguity, target disagreement and missing evidence, placeholder rejection after repairs, successful recovery after a corrected implementation, no paper verdict on exhaustion, repository prioritization, incomplete review wording, v3 cache invalidation, API classification and job-slot release. Frontend production build, tests and lint pass; browser checks cover target and method stops at desktop and mobile sizes, absence of a numeric verdict/certificate, and restored reasons in Catalog and Discovery.

Passing gates: 220 backend tests covering the loop, catalog, extraction, API, export and spending controls; 103 Discovery tests; 138 frontend tests. Four production-build browser scenarios at 1440×1000 and 390×844 passed without page errors or POST requests.

Two previously documented Discovery query-limit tests remain outside the passing selection: they expect three queries where unchanged production behavior permits five. No paid model, Tavily or sandbox calls were started for this repair.

This is a focused negative check for known retrieval placeholders, not a calibrated general method-fidelity verifier. A clean guard result does not certify SARAF or any other algorithm. The next actual run can still fail, time out, or miss the paper value; it must retain its measurements and limitations.

After deployment, use a fresh Discovery search before checking SARAF again. The existing 0.366 candidate retains its original identity and is stopped by the new target check; it is not converted into a successful reproduction. A faithful new SARAF reproduction remains untested.

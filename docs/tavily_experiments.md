# Tavily findings become experiments (role 3, stage 2)

Status: **design, awaiting approval** (2026-10-07). Nothing is built yet. Rules in this file are fixed before any real run and are not changed after seeing results.

## Goal

Today role 3 ("What others found", `community.py`) only reports: verified objections with a verbatim quote and URL, shown next to the verdict. This change turns each verified objection into one of four things, decided by the LLM once, then handled by fixed code:

| Type | Meaning | What runs | Result |
|---|---|---|---|
| `seed_variance` | "the number depends on the seed / is unstable" | extra seeds, **same code** as the main measurement, own branches | spread, verdict on the objection |
| `parameter_change` | "with setting X = B the result differs" (e.g. input window 336 vs 96) | code role changes **exactly that setting**, own branch, own harness measurement | delta to the main measurement, verdict |
| `reported_number` | "I measured value V (on setup S)" (e.g. issue #117: MSE 0.426 vs 0.386) | **no new branch**; link to our own measurement | which value our measurement supports |
| `not_measurable` | fairness, novelty, wording, questions, bugs of the original code | nothing | stays text |

Everything below is about **reporting**, as before: the main verdict (axis 1, axis 2, overall statement) is never changed by an objection check. The harness contract is unchanged (agent writes `predictions.csv`, our harness measures). Mode "Nachbau" is unchanged (the original repo never enters an agent branch).

Each checked objection keeps its verbatim quote and URL (existing rule: quote must occur in the page text, page must name the paper or be an issue page of the paper's repo).

## Where it runs in the loop

```
run(): ... all claims -> verdicts -> _community_round (findings, as today)
                                   -> _objection_round (NEW, only if community findings exist)
```

`_objection_round` runs after the last verdict, in the product path only (same gate as the community round: `graph_repair=True`, `use_tavily`, `tavily_mode == "live"`), plus an explicit switch `objection_experiments=True` on the Orchestrator (default off, so benchmark v2–v4 and all pre-registered experiments stay reproducible and unchanged). Failures become warnings (as in the community round); `BillingBlocked` passes through. Experiments never touch `dossier.verdicts` or `dossier.attempts`.

No new Tavily call: the round consumes `dossier.community["findings"]` (already paid, 7-day cache per arXiv ID). A fixture or cache run therefore costs 0 Tavily credits.

## Data model

New dossier field (default `[]`, old dossiers load unchanged):

```python
objection_checks: list[dict]   # one entry per checked finding
```

Entry (`dataclass ObjectionCheck` in new module `objection_experiments.py`, stored via `asdict`):

```python
id: str                    # "obj1", "obj2", ... (stable within a run)
claim_id: str              # the paper claim it is compared against
topic: str; statement: str; quote: str; url: str     # verbatim from the finding
kind: str                  # finding kind from community.classify (protocol note / ...), unchanged
type: Literal["seed_variance","parameter_change","reported_number","not_measurable"]
type_reason: str           # one sentence from the LLM (why this type)
# parameter_change
param: str | None          # key from the task allow-list, e.g. "input_window"
from_value: str | None     # our main setting, taken from the plan/code by the LLM or the allow-list default
to_value: str | None       # value the objection names; must occur in the quote
# reported_number
reported_value: float | None   # must occur in the quote as a number
reported_metric: str | None    # must equal claim.metric, else not_measurable
same_setup: Literal["yes","no","unknown"]
# result
status: Literal["bestaetigt","widerlegt","offen","nicht pruefbar"]
rule: str                  # the fixed rule text that was applied (for the report)
measured: dict | None      # {mean, std, n, seeds:[...], main_mean, main_std, delta, supports}
attempts: list[dict]       # {branch_id, seed, status, measured, code_hash, parent_snapshot}
code_hash: str | None
cost_usd: float
note: str                  # why "offen"/"nicht pruefbar" (task does not support the parameter, invalid run, ...)
```

Attempts of objection branches are stored **inside** the entry, not in `dossier.attempts`, so verdict, charts and branch tree of the main measurement do not change. The Evidence view draws them as a separate group.

## Events (events.jsonl, streamed unchanged via `on_event`/SSE)

- `objection_classified`: `{claim_id, id, type, topic, quote, url, param?, to_value?, reported_value?}` (one per finding)
- `objection_experiment`: `{id, type, stage: "start"|"seed"|"end", ...}` with `branch_id`, `seed`, `measured`, `status`, `code_hash` on `seed`
- `objection_checked`: `{id, claim_id, type, status, rule, measured, quote, url}` (final record, also what a replay needs)

`web/src/components/run/phases.ts` gets an "Objections" phase after the community phase; old runs without these events are unaffected.

## Classification (one LLM call per finding, role `extract`)

Input: paper title, dataset, the claim (text, metric, target, tolerance), the task's allow-list of changeable parameters (below), the finding (statement + quote). Output JSON:

```json
{"type": "seed_variance|parameter_change|reported_number|not_measurable",
 "reason": "one sentence",
 "param": "<allow-list key or null>",
 "to_value": "<value as written in the quote or null>",
 "reported_value": <number as written in the quote or null>,
 "reported_metric": "<metric name or null>",
 "same_setup": "yes|no|unknown"}
```

Prompt rules (fixed text, in `objection_experiments.CLASSIFY_PROMPT`):

1. Choose exactly one type. When in doubt choose `not_measurable`.
2. `seed_variance` only if the quote itself says that the result varies with seed / initialisation / run.
3. `parameter_change` only if the quote names a concrete setting **and** a concrete value, and the setting is in the allow-list for this task. Otherwise `not_measurable`.
4. `reported_number` only if the quote contains a number someone measured with the same metric as the claim. `same_setup` is `yes` only if the quote names the same dataset and horizon/split; `unknown` if it does not say.
5. Answer only from the text given; no invented values.

**Deterministic checks after the LLM (the LLM cannot override these):** `to_value` and `reported_value` must occur in the normalised quote (same `_norm` as the quote check); `param` must be in the allow-list; `reported_metric` must equal `claim.metric`; otherwise the type is demoted to `not_measurable` with a note. The LLM is asked once per finding at temperature 0; the result is cached in the run (no re-ask on replay).

## Which tasks support which parameter changes (allow-list)

The change is a *sentence in the code prompt*, "keep everything else". The task's data description fixes what the agent may vary; the harness (labels, splits, metric) stays ours.

| Task | `param` keys | Notes |
|---|---|---|
| `etth1` (DLinear, NLinear, ...) | `input_window` (L ≤ 720, the task allows any L ≤ 720), `epochs`, `learning_rate`, `batch_size`, `individual` (per-channel vs shared linear layer), `decomposition_kernel` | `input_window` is the only one with a clean public objection so far (336 vs 96, role 3 stage 1) |
| `cora` / `cora_lcc` (GCN, GAT, ...) | `hidden_units`, `dropout`, `weight_decay`, `learning_rate`, `epochs`, `num_layers`, `feature_normalization` | splits are fixed by the task and **not** changeable: objections about the split/protocol are `not_measurable` ("not supported by this task") |

Anything outside the list is `not_measurable` with note "setting not changeable in this task". The allow-lists live in `tasks.py` (`Task.tunable: tuple[str, ...]`, default `()` so unknown tasks run no `parameter_change`).

## Experiments and fixed verdict rules

`tol` = claim tolerance, `main_mean`/`main_std` = the claim's measured mean/std over its seeds (from the verdict). Let `S` = number of main seeds. All numbers are the harness metric of the claim. The rules below are fixed now.

### seed_variance
Run `K = min(5, S)` **extra** seeds `S .. S+K-1` with the **final main code** (the code of the verdict, `discarded` hashes excluded), one branch each, same pipeline as the main seeds (smoke, `unshare -n`, static code check, fresh harness). Pool = main seeds + extra seeds.
- `bestaetigt` if `std(pool) >= tol` **or** `|mean(extra) - main_mean| > tol`.
- `widerlegt` if both are false and at least `ceil(K/2)` extra seeds are valid.
- `offen` otherwise (fewer valid extra seeds, or the main run has no valid seeds).

Rationale: a seed dependence matters when the spread alone is as large as the tolerance used for the verdict, or when fresh seeds land elsewhere than the original seeds.

### parameter_change
One new code from the code role: prompt = task description + plan + final main code + "Change exactly this setting: `<param>` from `<from_value>` to `<to_value>`. Change nothing else. Return the full script." Then the normal seed-0 repair loop (`_run_seed`, max `MAX_ATTEMPTS_PER_CLAIM` attempts), then `K3 = min(3, S)` seeds total (seed 0 plus 2 more, seeds `0..K3-1`), own branches. Main comparison uses the **same seeds** of the main run (`main_mean_k3` = mean of the main measurement on seeds `0..K3-1`) so the comparison is paired.
- `delta = new_mean - main_mean_k3`.
- `bestaetigt` if `|delta| > tol` (the named setting moves the result by more than the verdict tolerance).
- `widerlegt` if `|delta| <= tol` and at least 2 valid seeds.
- `offen` if fewer than 2 valid seeds, the static check or smoke check fails after the repair attempts, or the code diff does not touch the named setting.
- Reported next to it (information, not part of the rule): whether `new_mean` is within `tol` of the paper target.

A cheap deterministic guard on "changes exactly that setting": the new code must contain the literal `to_value` (as number or word), else `offen` ("code did not apply the change"). Whether other settings changed too cannot be fully proven; the diff against the main code is stored (`code_hash` of both) and shown.

### reported_number
No branch. Compare the reported value `r` with our measurement `m = main_mean` and the paper target `t`:
- `bestaetigt` ("our measurement supports the reported value") if `same_setup == "yes"`, `|m - r| <= tol` and `|r - t| > tol`.
- `widerlegt` ("our measurement supports the paper") if `same_setup == "yes"`, `|m - t| <= tol` and `|m - r| > tol`.
- `offen` in all other cases (setup unknown or different, both within tolerance, neither within tolerance, no valid main measurement). `measured.supports` states `"reported"`, `"paper"`, `"both"` or `"neither"`.

Issue #117 example: reported 0.426 for `etth1 96 48 96` (input 96 / label 48 / horizon 96), paper 0.386 (that issue's number), claim target 0.375. The quote does not state the same setup as our task (input window), so the LLM must answer `same_setup: "unknown"`, giving `offen`. That is the honest outcome and the first test case.

### not_measurable
`status = "nicht pruefbar"`, stays text, shown with its reason.

## Limits (cost control inside a run)

- At most **3 experiments per run** (constant `MAX_OBJECTION_EXPERIMENTS = 3`); priority `parameter_change` > `seed_variance` > rest, ties by order in the report. `reported_number` and `not_measurable` cost no sandbox time and are unlimited.
- Both experiment types use the existing run cost guard (`MAX_COST_PER_RUN_USD`); if the budget is reached, remaining objections get `offen` with note "run budget reached".
- Duplicate objections (same type, param, to_value) run once; the others link to it.
- Sandbox: `seed_variance` ≤ 5 branches, `parameter_change` ≤ 3 branches plus repair attempts. Experiments of one run run sequentially (at most `MAX_PARALLEL` seeds in parallel inside one experiment, as for the main seeds).

## Cost and credit estimate per run

LLM (Token Factory, gross incl. 19 % VAT; prices from CLAUDE.md: Lightning 0.06/0.24, Super 0.30/0.90, Ultra 1/3 USD per million tokens):

| Step | Model | Estimate |
|---|---|---|
| Classification, ~6–10 findings × ~1.5k in / 0.5k out (with reasoning text maybe 3k out) | `extract` = Super | ~0.01–0.02 USD |
| `parameter_change` code, ≤ 3 per run × (~4k in / ~3k out) + ≤ 2 repairs each | `code` = Super | ~0.02–0.05 USD |
| `seed_variance`, `reported_number` | none | 0 |
| **Total** | | **≈ 0.03–0.07 USD net, ≈ 0.04–0.09 gross** |

The Kipf main run costs 0.074 USD; the check adds roughly the same order. Hard cap for the whole test (phase C): 1.00 USD gross.

Tavily: **0 additional credits** (the round reads the community findings). A live run's own Tavily use is unchanged: roles 2/4/5 plus the community round: Search basic 1 credit each (≤ 3), Crawl `limit=3` ≈ 2 credits (mapping 1 + extraction 1), Extract of README ≈ 1–2 credits (advanced ×2). Estimated ≈ 8–12 credits per live run by the `tavily_usage.py` estimator; two live runs ≈ 20–24 credits, under the hard cap of 60.

Sandbox (Nebius, free): seed_variance 5 × ~25 s (GCN) or ~5 × 1 min (DLinear); parameter_change 3 seeds + code attempts. Adds roughly 3–8 minutes to a run.

## Report and Evidence view

Zeugnis (`report.py`): new section `## Objections checked (Tavily role 3 → own experiments)` after "What others found": one line per entry: `"<quote>" (<url>) → <what was measured> → <status>`, e.g. `"…MSE 0.426, not in paper 0.386…" (…/issues/117) → reported number; our measurement 0.3769 ± 0.0021 supports the paper → widerlegt` or `→ input window 336 vs 96: MSE 0.3771 vs 0.3769 (Δ 0.0002, tolerance 0.010) → widerlegt`. Wording is careful: "bestaetigt/widerlegt" always means "by the rule fixed in advance on this task", the objection's author may have used another setup.

Web: `CommunityCard` gets a new block "Objections checked" (only when `dossier.objection_checks` is non-empty; old runs unchanged): per entry type badge, quote with link, measured value with spread and tolerance, status badge, and for experiments the branch ids (links to the branch tree group "objection branches"). Findings that are checked appear there instead of twice: the plain "What others found" list stays and gets a small "checked" pointer.

## Tests (phase B, fakes only)

Type classification with fake LLM (all four types, demotion by deterministic checks: value not in quote, param not in allow-list, wrong metric), each verdict rule at its boundaries, `offen` paths, limits (3 experiments, duplicates), run-budget stop, events written in order, dossier round trip with and without the new field, old dossiers load, verdicts untouched, `objection_experiments=False` runs nothing, fixture/offline mode runs nothing, report section, web build/lint/test.

## Pre-registered test (phase C, written before any run)

`benchmark/experiments/tavily_experiments_test.md`: (1) fixture run on DLinear using `role3_zeng_objections.json` (findings from the stored pages, LLM live is needed for extraction/classification; the question is whether to run it on Token Factory or with an LLM stub, see open questions); (2) after approval one live Kipf run and one live DLinear run; caps 1.00 USD gross, 60 Tavily credits.

## Open questions for the coordinator

1. **Kipf has no role-3 fixture** (`benchmark/fixtures/` has `role3_zeng_objections.json` only; the Kipf repo issue pages are in `logs/community_cache.json` only if a live run cached them). Proposal: no fixture run for Kipf; Kipf only live (phase C, ≈ 8–12 credits). OK?
2. The fixture test on DLinear needs a real LLM for extraction and classification (the stored pages are raw text). Proposal: run it on Token Factory (small: 5 pages) and spend the sandbox runs for real, as a dress rehearsal before the live runs. Alternative: only fake-based tests in phase B, no fixture run.
3. Rule thresholds above (`std >= tol`, `|delta| > tol`, K = 5/3) are my proposal; tell me if you want others before they are frozen.
4. Default of `objection_experiments`: off in the Orchestrator and the benchmark scripts, **on** in the API live path (`api.py`) once approved. OK?

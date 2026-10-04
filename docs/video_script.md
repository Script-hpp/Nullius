# Demo video script (under 3 minutes)

Status: script for the reviewer. Open decisions are marked `<!-- REVIEW: ... -->`. Benchmark numbers are from Benchmark v4 (`benchmark/experiments/results/benchmark_v4.md`, also `web/public/data/benchmark.json`); the numbers in scene 6 and 7 must not be changed without re-reading that file.

Message of the video: Nullius is a **high-precision checker**. When it says "target reached", an independent harness measured it, and it never accepted a deliberately wrong claim. Then, honestly: it often misses the target, mostly by a small margin.

Rules that apply (Official Rules, summarised in `CLAUDE.md`, section "Hackathon-Regeln"): public on YouTube, under 3 minutes, shows the project working, no protected music, no third-party marks without permission, English or English subtitles. Total runtime here: **2:46**, narration 297 words at 2.3 words per second (limit for the plan: 2:50).

## Changes against the three acts in `CLAUDE.md`

`CLAUDE.md` planned: (1) GCN on Cora, (2) DLinear beats the Transformer, (3) a recsys paper that fails. The actual state is different:

- **Recsys (Dacrema et al.) is not built.** On MovieLens-1M the paper itself reproduces NeuMF and finds it ahead of the simple baselines, so there is no negative case (`benchmark/experiments/benchmark_v2.md`, section "Dacrema: nicht gebaut"; `docs/reviewer_handover.md`, "Entscheidungen des Nutzers").
- **DLinear "beats the Transformer" is not a tested claim.** The DLinear claim checks one MSE value on ETTh1, not a ranking against other models (`benchmark/papers.yaml`). Do not say it in the video.
- The video therefore uses: **Act 1** a live run of a catalog paper (Tavily live calls, branches, harness), **Act 2** a positive certificate (GCN on Cora), **Act 3** the negative controls and the honest limits. See the candidates below.

## Candidates for the negative case (decision: reviewer)

All rows are existing runs. None is a real negative finding about a paper; each shows that Nullius does not pass a number it cannot measure.

| Candidate | Path | What it shows | Caveat |
|---|---|---|---|
| **A. Placebo claim on Cora** (decided by reviewer, final choice: user) | **Exported production run `web/public/data/kipf_gcn_2017/20261002-110746`** (`#/run/kipf_gcn_2017--20261002-110746`): real claim `gcn_cora_acc` (target 0.815, measured 0.8151 ± 0.0107, "reproduced") and control claim `gcn_cora_acc_placebo` (target 0.88, same measurement, "not reproduced"); run cost 0.139 $ gross (`run_end` event), 10 seeds. | A deliberately wrong target value is rejected by the harness, in the same run as the real claim. Simple to explain in one sentence. | Constructed claim, not a paper claim: say "control claim". Older fallback: `runs/kipf_gcn_2017/20260929-134108` (Benchmark v1, measured 0.7964), not needed now. <!-- REVIEW: reviewer chose A; user decides finally --> |
| **B. "Agent fault suspected"** | `web/public/data/disc_2601_21866/20260930-204237` (MoHETS on ETTh1, measured MSE 2.319 against target 0.35) | The baselines check flags a result far worse than a trivial baseline as an agent fault instead of blaming the paper. Already exported and visible at `#/run/disc_2601_21866--20260930-204237`. | README and benchmark text say this verdict says nothing about the paper. A newer catalog entry for MoHETS says "not reproducible (setup)" (`benchmark/catalog.yaml`, run `runs/disc_2601_21866/20260930-235615`, uncommitted). Use one of the two, not both. |
| **C. Stuart-Landau GNN (SL-GCN)** | exported live run `web/public/data/disc_2511_08094/20260930-221657` (quick check, 3 seeds, measured 0.787, target 0.829); newer catalog entry "agent fault suspected" (`benchmark/catalog.yaml`, run `runs/disc_2511_08094/20261001-000553`, mean 0.236) | Shows a live run on a fresh paper with a verdict that is honestly provisional. | Overall statement is "provisional" (axis 2 not checked); the LLM answers of the exported live run came from the cache (`cost_gross_usd` 0.0). Not a clean negative. |

Decision (reviewer, user confirms): **A** for Act 3. **B and C are not negative cases**; at most they appear as a short "honest limits" mention in scene 8. **C** stays usable for the live run in Act 1 (real Tavily calls in the event log).

## Scene list

Timings are start times. Narration is written to be spoken at a normal pace: about 2.3 words per second, so a 15 second scene holds at most 34 words. Word counts are in the last column.

| # | Time | On screen | Narration (spoken, English) | Footage needed | Words |
|---|---|---|---|---|---|
| 1 | 0:00-0:16 | **Hook.** Landing page, "The paper reports 81.5%" (Kipf & Welling, GCN on Cora). Hold on the number for the first 3 seconds, then the tagline "Other agents fix code until the tests pass. Here, the paper is the test suite." | "A paper claims its GCN reaches 81.5 percent on Cora. Do we believe it? Nullius in verba: take nobody's word. Let's check. Other agents fix code until the tests pass. Here, the paper is the test suite." | Screen recording of the landing page (`#/`), claim block `#claim`, light theme. | 37 |
| 2 | 0:16-0:30 | **Discover** (`#/discover`): run Tavily search, list of found papers with verbatim quote and "protocol matches / unclear" badges. | "Tavily searches the web live for new papers on benchmarks we support, reads only protocol and results, and proposes a claim with a verbatim quote. No matching quote, no claim." | Recording of **one real live discovery run** (one real Tavily call, visible for the Tavily prize; reviewer makes it). Fallback only: the stored replay `web/public/data/discovery/replay.jsonl`, labelled "replay" on screen. <!-- REVIEW: live discovery costs Tavily credits; reviewer chose live, user decides finally --> | 30 |
| 3 | 0:30-0:40 | **Catalog** (`#/catalog`): cards with claimed value, measured value, seeds, tolerance, cost. Cursor on one card, "Run live" button, access-code field (code not readable). | "Every catalog paper has already run once through the full loop. Pick one, enter the access code, start a live check." | Recording of the catalog page; type the code off-screen or blur the field. | 21 |
| 4 | 0:40-1:16 | **Live run** (`#/live/<job>`): phase timeline. Tavily badges "live" for role 2 (paper text) and role 4 (README); plan; code; branch tree growing seed by seed; measurements arriving; the duration hint ("usually about ...") visible. | "Now it runs live. Tavily extracts the paper and the README, every call marked live with a hash. Nemotron Ultra plans, Nemotron Super writes the code, Nemotron Lightning reads the tables. The code never runs on our server: each seed is a branch of one Nebius Sandbox snapshot, without network. A failed attempt is rolled back and repaired." | Screen recording of one real live quick check (3 seeds) on Token Factory. Cut the waiting time, keep the timer visible and say so on screen ("sped up"). Needs the API on `:8000` with `LLM_PROVIDER=tokenfactory`, `LLM_CACHE` off (the API sets it). | 58 |
| 5 | 1:16-1:34 | **Evidence** (`#/run/kipf_gcn_2017--20261002-110746`), real claim: target 81.5% with its Tavily source on the left, sandbox measurement 81.5% ± 1.1 on the right, branch tree. Caption: "target reached (method fidelity not checked)". | "The agent never reports a number, only predictions. A harness we own loads the test labels and computes the metric. Mean over all seeds, tolerance fixed in advance: target reached, with spread." | Recording of the same exported run (or a newer production run, reviewer decides). | 32 |
| 6 | 1:34-2:00 | **Climax: the control claim.** Same Evidence page, second claim `gcn_cora_acc_placebo`: target 88%, measured 81.5%, badge "not reproduced". Then a flash of the Benchmark page counter "0 of 48". | "Now the test that matters. We give it a wrong target: 88 percent on Cora. The harness measures about 81 and says not reproduced. We ran this control in every benchmark run: forty-eight wrong claims, none reported as reproduced." | Recording of the control claim in the exported run; one second of the Benchmark page (`#benchmark`) with the placebo number. | 39 |
| 7 | 2:00-2:24 | **Benchmark** page (`#benchmark`) or the README table: three numbers (0 of 48 wrong claims reproduced; 7 of 7 "reproduced" verdicts agree with the reference; target reached for 2 of 8 claims with Tavily, 1 of 8 without). Caption: "v4, 48 runs, method fidelity not checked". | "Six papers, eight claims, three fresh generations each, with and without Tavily. Whenever Nullius said reproduced for a real claim, our reference agreed: seven of seven. But it often misses. The target was reached for two of eight claims with Tavily, one without, mostly by a small margin." | Screen recording of the Benchmark page (latest evaluation v4) or a screenshot of the README section "Benchmark". Numbers only from `benchmark/experiments/results/benchmark_v4.md`. | 48 |
| 8 | 2:24-2:46 | Closing slide: honest limits, then the repository URL and the licence. List: "Method fidelity: not checked", "Supported: Cora, Cora with random splits, ETTh1", "CPU only", "No real negative case: controls are constructed". | "Limits, stated up front. Method fidelity is not checked, so reproduced means target value reached. Cora and ETTh1 only, CPU only. Code is MIT licensed. Nemotron on Token Factory, Nebius Sandboxes, Tavily." | Static slide (text only, no third-party logos unless permission is confirmed; the plain names are enough). | 32 |

Sum: 2:46, 297 words (about 129 seconds of speech; the rest is screen time). The last 4 seconds to 2:50 stay free as a safety margin. "Reproduced" in scenes 6 and 7 is said together with "method fidelity is not checked" on screen (caption) and in scene 8 (voice-over).

## Must-show checklist (from the brief)

| Must show | Scene |
|---|---|
| Nemotron roles on Token Factory (Lightning, Super, Ultra) | 4 (named), 8 |
| Nebius Sandbox branches | 4 (branch tree, one branch per seed) |
| Tavily live calls at run time (required for the Tavily prize) | 2 (discovery), 4 (role 2 and 4 badges "live"), evidence page event log in 5 |
| Independent harness: agent delivers only predictions | 5 |
| Negative control (placebo claim) as the climax | 6 |
| Benchmark numbers (v4: 0 of 48, 7 of 7, 2 of 8 / 1 of 8) | 6 (0 of 48 on screen), 7 |
| Honest limits (axis 2 "method fidelity not checked", supported datasets only) | 7 (caveat line), 8 |

Wording rule from `benchmark/experiments/benchmark_v2.md`: the word "reproduced" never stands without "method fidelity not checked" in written material. In the voice-over, scenes 5 and 7 say "target reached" or "reproduced" and scene 8 carries the caveat; the on-screen caption of scene 5 must read "reproduced (target value reached; method fidelity not checked)".

## Recording checklist

Prepare before recording:

- [x] Benchmark v4 finished and evaluated (`benchmark/experiments/results/benchmark_v4.md`, `web/public/data/benchmark.json`); the README section "Benchmark" carries the same numbers. Record scene 7 from the Benchmark page or the README.
- [ ] API running with production settings: `LLM_PROVIDER=tokenfactory`, `SANDBOX_BACKEND=nebius`, `TAVILY_MODE=live`, an `ACCESS_CODE` set; budget left in `BUDGET_TOTAL_USD` (check `scripts/cost_report.py`). Web app from the same origin (Vite `:5174` or the built app via `deploy/serve.py`).
- [ ] One live quick check (3 seeds) on a catalog paper finished **before** recording, so a fallback exists: the Live page switches to the replay of the last run if a live run cannot start (README, "Product tour").
- [ ] Catalog page shows the chosen paper with a finished result (catalog entries and exports are the reviewer's uncommitted files until committed).
- [x] Negative-control run (candidate A) exported and visible on the Evidence page: `kipf_gcn_2017--20261002-110746` (production run, 0.139 $ gross). Check that the control claim is shown below the real claim and the page opens without errors.
- [ ] Evidence page for scene 5 opens without errors, Tavily source and branch tree visible, tab "replay" works.
- [ ] One real live discovery run recorded for scene 2 (at most once, Tavily credits); replay file only as fallback.
- [ ] Both themes checked; record in one theme only (light recommended for contrast).

Keep out of the picture:

- [ ] No API keys, no `.env`, no terminal with environment variables. The access code must not be readable (blur, or type it off-screen).
- [ ] No user names in paths: hide the home directory in terminals and browser bars; record the browser only, not the file manager. Scene 4 `code.py` view shows agent code only, no paths.
- [ ] No Token Factory, Tavily or Nebius console pages with account data (balance, project ids, email).
- [ ] No bookmarks bar, other tabs, notifications.
- [ ] Disable desktop notifications and screen lock.

Post-production:

- [ ] Speed-up labelled on screen ("sped up") wherever the live wait is cut.
- [ ] Captions in English (the voice-over is English; captions help the jury read the numbers).
- [ ] Music: none, or a track with a licence that allows use; no protected music.
- [ ] Length below 3:00 in the upload (check the file, not the plan), public on YouTube, link in the Devpost form (`docs/devpost_submission.md`).
- [ ] Numbers in scenes 6 and 7 copied from `benchmark/experiments/results/benchmark_v4.md`, nothing from prototype runs on free endpoints (`CLAUDE.md`: prototype numbers are never published).

## Decisions (reviewer, 2026-10-01; the user decides finally)

1. Negative case: candidate A (placebo). Placeholder "placebo run from Benchmark v2 (reviewer picks and exports)", fallback run 134108. B and C only as a short "honest limits" mention.
2. Demo access: access code, as in the README; the code stays `TBD` in the test notes (`docs/devpost_submission.md`).
3. Scene 2: live discovery with one real Tavily call; replay only as fallback.

Status 4 Oct 2026 (decisions above unchanged): decision 1 is **done** (the control claim exists in the exported production run `kipf_gcn_2017--20261002-110746`; used in scenes 5 and 6); decisions 2 and 3 are **open until recording** (access code value, one real discovery run).

Still open:

- Scene 5: exported run from 29 Sep (`kipf_gcn_2017--20260929-130133`) or a fresh production run? <!-- REVIEW: default is the existing exported run -->
- Scene 7: **done**. The v4 result is weak on recall (2 of 8, 1 of 8), and the narration states that plainly next to the precision result. Update the scene only if the benchmark is re-run.
- Scene 5 now uses the exported run from 2 Oct (`kipf_gcn_2017--20261002-110746`), not the run from 29 Sep, because scenes 5 and 6 share one page; the user confirms.

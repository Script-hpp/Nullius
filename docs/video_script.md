# Demo video: current revision brief and historical script

## Current revision brief — 2026-10-10

The next cut follows [jury_walkthrough_and_video_next_steps.md](jury_walkthrough_and_video_next_steps.md), targeting about 2:30. The existing rendered MP4 has not yet been updated. The historical scene list below is not the current recording brief.

**Quality of the Idea:** make the central idea visible in the first 15 seconds and demonstrate it in the experiment sequence. A paper's claim becomes the acceptance criterion for an independently measured experiment.

Suggested opening voice-over:

> Before your team builds on a machine learning paper, can you reach its reported result? Nullius turns a paper's claim into a test. Here, the paper is the test suite.

Suggested voice-over during source, generated code and measurement:

> Nemotron, running on Nebius Token Factory, reads the paper and the authors' code, then writes its own experiment. Nebius Sandboxes run it. Our independent checker measures its predictions against a target fixed before the run.

On screen: **Paper claim → own experiment → independent measurement → evidence certificate.** Show the source, the generated experiment and the actual measurements in that order. Use one recorded run; label it as recorded and label shortened waiting time. If showing a repair, use an actual documented failure and repair from that run.

Tie the demonstrated result to a practical decision: whether a researcher or ML team should investigate the paper further before building on it. Show the PDF certificate as a portable record of the result, setup and evidence. State inference costs as inference costs; do not claim measured time savings or general success rates without evidence.

Evidence for the new cut: TSMixer and SparseTSF are post-development, manually selected examples; 7/8 refers to development units matching the reference verdict. Keep these separate and keep method fidelity marked as unchecked. GLinear has now reached its fixed target: MSE 0.3796 +/- 0.0003 versus 0.3820, three seeds, 0.205 dollars gross inference, preliminary. See [paper_candidates_20261010.md](paper_candidates_20261010.md) for the run and limits. Its website export is still pending; keep it out of recorded website scenes until evidence is accessible there.

**User decisions:** one person will test the product at the end. Wait for the test results before deciding the next live-demo steps. Later outreach to authors is an idea for presenting the project and its concrete reproduction findings; no emails have been commissioned or sent, and no email-draft button belongs in the product.

Official guidance: [four equally weighted criteria](https://nebiusglobalaihackathon.devpost.com/updates/46204-here-s-how-judging-works), [pitch and explicit Nebius/NVIDIA roles in audio](https://nebiusglobalaihackathon.devpost.com/updates/46205-how-to-build-a-winning-project).

## Historical script (preserved for reference)

Superseded by the rendered film (reviewer worktree, `video/`); this script is historical.

Status: draft for the reviewer. Open decisions are marked `<!-- REVIEW: ... -->`. Every number marked `TBD` depends on Benchmark v2 (`benchmark/experiments/benchmark_v2.md`) and must be filled from `benchmark/experiments/results/benchmark_v2.md` after the run, never from partial results.

Rules that apply (Official Rules, summarised in `CLAUDE.md`, section "Hackathon-Regeln"): public on YouTube, under 3 minutes, shows the project working, no protected music, no third-party marks without permission, English or English subtitles. Total runtime here: **2:45**.

## Changes against the three acts in `CLAUDE.md`

`CLAUDE.md` planned: (1) GCN on Cora, (2) DLinear beats the Transformer, (3) a recsys paper that fails. The actual state is different:

- **Recsys (Dacrema et al.) is not built.** On MovieLens-1M the paper itself reproduces NeuMF and finds it ahead of the simple baselines, so there is no negative case (`benchmark/experiments/benchmark_v2.md`, section "Dacrema: nicht gebaut"; `docs/reviewer_handover.md`, "Entscheidungen des Nutzers").
- **DLinear "beats the Transformer" is not a tested claim.** The DLinear claim checks one MSE value on ETTh1, not a ranking against other models (`benchmark/papers.yaml`). Do not say it in the video.
- The video therefore uses: **Act 1** a live run of a catalog paper (Tavily live calls, branches, harness), **Act 2** a positive certificate (GCN on Cora), **Act 3** the negative controls and the honest limits. See the candidates below.

## Candidates for the negative case (decision: reviewer)

All rows are existing runs. None is a real negative finding about a paper; each shows that Nullius does not pass a number it cannot measure.

| Candidate | Path | What it shows | Caveat |
|---|---|---|---|
| **A. Placebo claim on Cora** (decided by reviewer, final choice: user) | Placeholder: **placebo run from Benchmark v2 (reviewer picks and exports)**. Fallback: `runs/kipf_gcn_2017/20260929-134108` (claim `gcn_cora_acc_placebo`, target 0.88, measured mean 0.7964, verdict "not reproduced"; Token Factory run per `logs/costs.jsonl`, but Benchmark v1) | A deliberately wrong target value is rejected by the harness. Simple to explain in one sentence. | Not exported to `web/public/data` yet (needs `scripts/export_site_data.py`, reviewer). The fallback run is from Benchmark v1 (product configuration off). Constructed claim, not a paper claim: say "control claim". The same run also holds the real claim with verdict "partially" (0.7974), which would show on the page. Newer placebo runs exist in `runs/benchmark_v2.jsonl` once the benchmark finishes. <!-- REVIEW: reviewer chose A; user decides finally --> |
| **B. "Agent fault suspected"** | `web/public/data/disc_2601_21866/20260930-204237` (MoHETS on ETTh1, measured MSE 2.319 against target 0.35) | The baselines check flags a result far worse than a trivial baseline as an agent fault instead of blaming the paper. Already exported and visible at `#/run/disc_2601_21866--20260930-204237`. | README and benchmark text say this verdict says nothing about the paper. A newer catalog entry for MoHETS says "not reproducible (setup)" (`benchmark/catalog.yaml`, run `runs/disc_2601_21866/20260930-235615`, uncommitted). Use one of the two, not both. |
| **C. Stuart-Landau GNN (SL-GCN)** | exported live run `web/public/data/disc_2511_08094/20260930-221657` (quick check, 3 seeds, measured 0.787, target 0.829); newer catalog entry "agent fault suspected" (`benchmark/catalog.yaml`, run `runs/disc_2511_08094/20261001-000553`, mean 0.236) | Shows a live run on a fresh paper with a verdict that is honestly provisional. | Overall statement is "provisional" (axis 2 not checked); the LLM answers of the exported live run came from the cache (`cost_gross_usd` 0.0). Not a clean negative. |

Decision (reviewer, user confirms): **A** for Act 3. **B and C are not negative cases**; at most they appear as a short "honest limits" mention in scene 8. **C** stays usable for the live run in Act 1 (real Tavily calls in the event log).

## Scene list

Timings are start times. Narration is written to be spoken at a normal pace: about 2.3 words per second, so a 15 second scene holds at most 35 words.

| # | Time | On screen | Narration (spoken, English) | Footage needed |
|---|---|---|---|---|
| 1 | 0:00-0:14 | **Overview** (`#/`): hero line "Other agents fix code until the tests pass. Here, the paper is the test suite." and the list of finished certificates. | "Many machine learning results are hard to reproduce. Other agents fix code until tests pass. Nullius takes one claim from a paper and measures it. Nullius in verba: take nobody's word for it." | Screen recording of the Overview page, light theme, two seconds hold on the tagline. |
| 2 | 0:14-0:32 | **Discover** (`#/discover`): run Tavily search, list of found papers with verbatim quote and "protocol matches / unclear" badges. | "Tavily searches the web at run time for new papers on benchmarks we support, reads only the results and protocol sections, and proposes a claim with a verbatim quote. A claim without a matching quote is dropped." | Recording of **one real live discovery run** (one real Tavily call, visible for the Tavily prize; reviewer makes it). Fallback only: the stored replay `web/public/data/discovery/replay.jsonl`, labelled "replay" on screen. <!-- REVIEW: live discovery costs Tavily credits; reviewer chose live, user decides finally --> |
| 3 | 0:32-0:44 | **Catalog** (`#/catalog`): cards with claimed value, measured value, seeds, tolerance, cost. Cursor on one card, then one click on "Run live" (no sign-in, no code). | "Each candidate has already run once. Start a live check with one click. At the budget floor or the daily limit, live checks pause and the recorded run is shown." | Recording of the catalog page and the click. Optional 2-second cut-in of the paused state (`GET /api/status` with `live: false` shows the recorded-run notice), only if it can be recorded without stopping the real demo. |
| 4 | 0:44-1:22 | **Live run** (`#/live/<job>`): phase timeline. Tavily badges "live" for role 2 (paper text) and role 4 (README); plan; code; branch tree growing seed by seed; measurements arriving. | "Now it runs live. Tavily extracts the paper and the repository README, and every call is marked live with a hash. Nemotron Super writes the plan and the code. Nemotron Lightning reads the tables. The code never runs on our server: each seed is a branch of one Nebius Sandbox snapshot, without network access. A failed attempt is rolled back and repaired." | Screen recording of one real live quick check (3 seeds) on Token Factory. Cut the waiting time, keep the timer visible and say so on screen ("sped up"). Needs the API on `:8000` with `LLM_PROVIDER=tokenfactory`, `LLM_CACHE` off (the API sets it). |
| 5 | 1:22-1:42 | **Evidence** (`#/run/<id>`) for GCN on Cora: target value with Tavily source on the left, sandbox measurement with spread on the right, branch tree. | "The agent never reports a number. It delivers predictions. A harness we own loads the test labels itself, checks the predictions, and computes the metric. The verdict uses the mean over all seeds against a tolerance fixed before the run: here reproduced, with mean and spread." | Recording of `#/run/kipf_gcn_2017--20260929-130133` (measured 0.8211 ± 0.0072, target 0.815, verdict "reproduced"; Tavily role 4 call on `github.com/tkipf/gcn` in its event log). Or a newer production run, reviewer decides. |
| 6 | 1:42-2:02 | **Evidence** of the negative control (candidate A) or the agent-fault case (B). Verdict badge "not reproduced" with target 0.88 against measured 0.796. | "A control: we gave it a wrong target value, 88 percent on Cora. The harness measures about 80 and says not reproduced. A wrong claim does not pass. When a result is far below a trivial baseline, Nullius says agent fault suspected, not paper wrong." | Recording of the chosen page. For candidate A the placebo run from Benchmark v2 (or the fallback run) must be exported to `web/public/data` first. |
| 7 | 2:02-2:25 | **Benchmark section** of the README or a slide with the table. Numbers from Benchmark v4 (`benchmark/experiments/results/benchmark_v4.md`). | "We tested Nullius against known results: six papers, eight claims, each with and without Tavily, three fresh generations each. With Tavily, 2 of 8 claims matched the reference verdict. Without Tavily, 1 of 8. Wrong control claims reported as reproduced: none of 48." | Screenshot of the final table from README after Benchmark v2 (`benchmark/experiments/results/benchmark_v2.md`). Must not be recorded before the evaluation script ran on all 48 runs. |
| 8 | 2:25-2:45 | Closing slide: honest limits, then the repository URL and the licence. List: "Method fidelity: not checked", "Supported: Cora, Cora with random splits, ETTh1", "CPU only". | "Limits, stated up front. Method fidelity is not checked: our checker failed calibration three times, so a reproduced verdict means the target value was reached, not that the method was reproduced. Nullius supports Cora and ETTh1 today. Code is MIT licensed. Nemotron on Token Factory, Nebius Sandboxes, Tavily." | Static slide (text only, no third-party logos unless permission is confirmed; the plain names are enough). |

Sum: 2:45. The last 5 seconds to 2:50 stay free as a safety margin.

## Must-show checklist (from the brief)

| Must show | Scene |
|---|---|
| Nemotron roles on Token Factory (Lightning, Super) | 4 (named), 8 |
| Nebius Sandbox branches | 4 (branch tree, one branch per seed) |
| Tavily live calls at run time (required for the Tavily prize) | 2 (discovery), 4 (role 2 and 4 badges "live"), evidence page event log in 5 |
| Independent harness: agent delivers only predictions | 5 |
| Benchmark number (v4) | 7 |
| Honest limits (axis 2 "method fidelity not checked", supported datasets only) | 7 (caveat line), 8 |

Wording rule from `benchmark/experiments/benchmark_v2.md`: the word "reproduced" never stands without "method fidelity not checked" in written material. In the voice-over, scene 5 says "target value reached" once and scene 8 carries the caveat; the on-screen caption of scene 5 must read "reproduced (target value reached; method fidelity not checked)".

## Recording checklist

Prepare before recording:

- [ ] Benchmark v2 finished, evaluation written (`uv run python scripts/eval_benchmark_v2.py > benchmark/experiments/results/benchmark_v2.md`), README table filled, then scene 7 recorded.
- [ ] API running with production settings: `LLM_PROVIDER=tokenfactory`, `SANDBOX_BACKEND=nebius`, `TAVILY_MODE=live`, no access code (the demo is open); budget left above `LIVE_MIN_BUDGET_USD` and Tavily credits above `LIVE_MIN_TAVILY_CREDITS` (check `GET /api/status` shows `live: true`, and `scripts/cost_report.py`). Web app from the same origin (Vite `:5174` or the built app via `deploy/serve.py`).
- [ ] One live quick check (3 seeds) on a catalog paper finished **before** recording, so a fallback exists: the Live page switches to the replay of the last run if a live run cannot start (README, "Product tour").
- [ ] Catalog page shows the chosen paper with a finished result (catalog entries and exports are the reviewer's uncommitted files until committed).
- [ ] Negative-control run (candidate A) exported with `scripts/export_site_data.py` and visible on the Evidence page; only production runs are exported.
- [ ] Evidence page for scene 5 opens without errors, Tavily source and branch tree visible, tab "replay" works.
- [ ] One real live discovery run recorded for scene 2 (at most once, Tavily credits); replay file only as fallback.
- [ ] Both themes checked; record in one theme only (light recommended for contrast).

Keep out of the picture:

- [ ] No API keys, no `.env`, no terminal with environment variables.
- [ ] No user names in paths: hide the home directory in terminals and browser bars; record the browser only, not the file manager. Scene 4 `code.py` view shows agent code only, no paths.
- [ ] No Token Factory, Tavily or Nebius console pages with account data (balance, project ids, email).
- [ ] No bookmarks bar, other tabs, notifications.
- [ ] Disable desktop notifications and screen lock.

Post-production:

- [ ] Speed-up labelled on screen ("sped up") wherever the live wait is cut.
- [ ] Captions in English (the voice-over is English; captions help the jury read the numbers).
- [ ] Music: none, or a track with a licence that allows use; no protected music.
- [ ] Length below 3:00 in the upload (check the file, not the plan), public on YouTube, link in the Devpost form (`docs/devpost_submission.md`).
- [ ] Numbers in scene 7 copied from `benchmark/experiments/results/benchmark_v4.md`, nothing from prototype runs on free endpoints (`CLAUDE.md`: prototype numbers are never published).

## Decisions (reviewer, 2026-10-01; the user decides finally)

1. Negative case: candidate A (placebo). Placeholder "placebo run from Benchmark v2 (reviewer picks and exports)", fallback run 134108. B and C only as a short "honest limits" mention.
2. Demo access: **open, no access code** (decision of the user, 2026-10-05; `docs/devpost_submission.md`). Live checks pause at the budget floor or the daily limit and show the recorded run. Film v8 still shows the access-code field in scene 3 and must be re-rendered from this script.
3. Scene 2: live discovery with one real Tavily call; replay only as fallback.

Still open:

- Scene 5: exported run from 29 Sep (`kipf_gcn_2017--20260929-130133`) or a fresh production run? <!-- REVIEW: default is the existing exported run -->
- Scene 7 depends on the Benchmark v2 outcome. If the result is weak, the narration must state that plainly; the script keeps `TBD` until then.

# Devpost submission (17 Oct 2026)

Copy-ready text for the Devpost form, in the order of the form fields. Numbers match the README and the stored certificates in `web/public/data/`. Rules for the wording: plain words first, “target reached with our setup”, never “verified”; method fidelity is not checked; no general success rate.

---

## Project name

Nullius

## Elevator pitch (max. 200 characters)

Papers make claims. Nullius measures them: an AI agent rebuilds an ML paper's experiment in a sandbox, and an independent checker tests whether the reported number holds.

## Try it out (links)

- Demo: https://nullius.akca.dev
- Code: https://github.com/Script-hpp/Nullius

## Video demo link

https://www.youtube.com/watch?v=EaXo1DadUkU

## About the project

### Inspiration

*Nullius in verba* is the motto of the Royal Society: take nobody's word for it. In a 2016 *Nature* survey of 1,576 researchers, more than 70% said they had tried and failed to reproduce another scientist's experiment. Checking one machine-learning claim by hand can take days.

Coding agents fix code until the tests pass. We wanted an agent whose test is a claim from a paper: **here, the paper is the test suite.**

### What it does

A paper says, for example, “our model reaches an error of 0.372 on ETTh1”. Nullius:

1. **Reads** the paper, the table with the number, and the authors' code, with word-for-word quotes as evidence.
2. **Rebuilds** the experiment: NVIDIA Nemotron writes its own code. The authors' program is never run.
3. **Measures** it in Nebius Sandboxes. The agent only delivers predictions; an independent checker loads the true answers and computes the score, over several seeds.
4. **Shows the verdict with the evidence**: what the paper claims, what we measured, and every source, code version and attempt behind it. The certificate can be saved as a PDF.

To make sure Nullius cannot simply agree with every paper, we plant a deliberately false claim next to the real ones. In our benchmark, **none of 48 planted false claims was accepted**.

Three checks you can open, or start fresh, in the demo:

| Paper | Paper claims | Nullius measured | Seeds |
|---|---:|---:|---:|
| FITS (forecast error, ETTh1) | 0.372 | 0.37203 ± 0.00005 | 3 |
| Kipf & Welling GCN (accuracy, Cora) | 0.815 | 0.8146 ± 0.0077 | 9 |
| GLinear (forecast error, ETTh1) | 0.3820 | 0.3796 ± 0.0003 | 3 |

FITS is the strongest case: it ran on the public server without human edits, and all three seeds landed within 0.0001 of the claim, for $0.31 of inference. “Reproduced” here means the target number was reached with our setup; the method itself is not verified. These are examples, not a success rate.

### What checking real papers taught us

The number in a paper is often not enough to rebuild it. Nullius surfaces this as findings (https://nullius.akca.dev/#findings):

- **FITS:** the results table lists four input lengths; the authors' launch script uses 360. With the script's settings our error matched the paper; with other settings it nearly doubled.
- **SparseTSF:** since a code change in March 2026, the authors' launch scripts for the linear model silently train the MLP variant. We reported it to the authors: https://github.com/lss-1138/SparseTSF/issues/16
- **RLinear:** a key setting appears only in a table caption. Our run missed it and did not reach the target; we show that miss openly.
- **TSMixer:** a matching number can come from a smaller model than the paper's. We label that “target reached with a reduced model”, not “reproduced”.

### How we built it

- **NVIDIA Nemotron on Nebius Token Factory** is the brain. Nemotron 3 Super plans the experiment, writes and repairs the code, and extracts setup details with quotes that must appear in the source. Nemotron 3.5 Lightning reads target values from tables. Nemotron never runs code.
- **Nebius Sandboxes** are the lab. One snapshot holds Python, the data and the checker; every attempt and every seed branches from it and runs in parallel. Training runs without network access.
- **Tavily** is the researcher, called live at run time: it discovers new papers on datasets we support, extracts paper text, READMEs and configs, and searches error messages when code fails. Every source is stored with a timestamp and a SHA-256 hash.
- **An independent checker** grades the predictions. The agent never sees the test answers and never grades itself.
- **A readiness check** runs every step before the first paid one, so a check ends in a verdict or in a clear stop with a reason, never in an unexplained crash.
- **The product:** Python orchestrator and FastAPI with live streaming (Server-Sent Events), and a React app with Discover, Catalog, Live run, Evidence, Findings and Benchmark pages. When the daily live budget is used up, the site plays a labelled recording of a real run instead.

### Challenges we ran into

- **Papers leave things out.** The decisive settings often sit in launch scripts, captions or code defaults, not in the paper's tables. We had to teach Nullius to read them and to stop with a reason when they contradict the claim.
- **Telling our mistakes from the paper's.** A miss can be an agent bug. Trivial baselines, smoke tests, ablations and recorded repairs help separate the two, and failed checks stay visible.
- **Honest limits.** Our automatic check of whether the code really implements the paper's method missed its calibration targets three times, so we label that dimension “not checked” everywhere instead of claiming it.
- **Platform surprises.** A model was removed mid-benchmark, and sandbox output is cut at 64 KB. Details and evidence are in our Nebius feedback.

### Accomplishments that we're proud of

- An autonomous live run that hit the FITS number to four decimal places, with every source and attempt inspectable.
- 0 of 48 planted false claims accepted.
- A real bug report to paper authors, found by reading their code.
- Every number on the site and in this text links to a stored record in the repository.

### What we learned

Rebuilding from the paper alone is often not enough; reading the authors' code helped most (7 of 8 development cases matched our reference verdict, measured in-sample; a separate held-out paper missed). And a careful “not checked” is worth more than a confident wrong verdict.

### What's next

More supported datasets, a calibrated check of method fidelity, and turning community findings into their own experiments. We shared our results with the authors of FITS and GLinear and are waiting for their feedback.

## Built with

python, fastapi, react, typescript, vite, tailwindcss, nvidia-nemotron, nebius-token-factory, nebius-sandboxes, tavily, pytorch, numpy, docker, server-sent-events

## Testing instructions for judges

The demo is open, no sign-in.

1. Open https://nullius.akca.dev and press **Watch a check in 30 seconds** for a recorded replay, no waiting.
2. Open **Catalog** and pick **FITS**, **Kipf GCN on Cora** or **GLinear**. Open the finished result: claim, measurement, sources, generated code, per-seed values, PDF export.
3. To start a fresh check, press the start button on one of these three. A readiness check runs first; the run then takes several minutes and makes paid calls. Live checks are limited per day; when the limit is reached, the page explains why and offers the recorded run.
4. **Discover** lets Tavily search for new papers on supported datasets (limited per day as well).
5. **Findings** and **Benchmark** show what we learned and what we measured, including our misses.

## Feedback on Nebius Token Factory, AI Cloud and NVIDIA (form field)

Use the “Devpost text field” section at the end of [nebius_feedback.md](nebius_feedback.md); the full feedback with evidence is linked from there.

## Gallery images

Screenshots for the gallery (3:2, taken from the live site): `docs/img/devpost/`.

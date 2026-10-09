import { useEffect, useState, type ReactNode } from 'react'
import { ArrowRight, Bug, ChevronDown, ExternalLink, FileText, Quote, Search, Users } from 'lucide-react'
import { FIGURES, HELD_OUT, LIMITS, README_TAVILY, REPO_BLOB, SCOPE_LIMIT, SOURCES } from '../data/benchmark'
import { FEATURED_RUN_ID } from '../featured'
import { loadIndex, loadRun, type RunData } from '../data'
import { showValue } from '../format'
import { RESULT, de2en, resultHint, translateClaim } from '../labels'
import { useLiveStatus } from '../live'
import { statusFor } from '../livestatus'
import { Matrix } from './TwoModes'
import { Badge } from './ui'

// The sections of the start page after the hero. They read as one story while scrolling: problem, how it works,
// one real certificate, Tavily, proof and lesson, limits, try it. One thought per section, at most one link.
// Every benchmark number is read from data/benchmark.ts; the certificate numbers come from the exported run.

export const EXAMPLE_RUN = `#/run/${FEATURED_RUN_ID}`
export const TIMELAPSE_RUN = `#/run/${FEATURED_RUN_ID}?play=1`

const ext = { target: '_blank', rel: 'noreferrer' } as const
const link = 'inline-flex items-center gap-1.5 text-fg underline underline-offset-4 hover:text-fg-max'
const lead = 'font-grotesk m-0 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl'
const body = 'm-0 max-w-2xl text-[16px] leading-relaxed text-fg-4'

export function Block({ id, label, children }: { id?: string; label: string; children: ReactNode }) {
  return (
    <section id={id} aria-label={label} className="mx-auto max-w-4xl scroll-mt-8 px-6 py-16 sm:py-24">
      {children}
    </section>
  )
}

// The exported example run, loaded once; the numbers on the page come from its dossier, not from this file.
let featured: Promise<RunData | null> | null = null
function loadFeatured(): Promise<RunData | null> {
  featured ??= loadIndex().then((runs) => {
    const ref = runs.find((r) => r.id === FEATURED_RUN_ID)
    return ref ? loadRun(ref) : null
  }).catch(() => null)
  return featured
}
export function useFeatured(): RunData | null {
  const [run, setRun] = useState<RunData | null>(null)
  useEffect(() => { loadFeatured().then(setRun) }, [])
  return run
}

export function Problem({ run }: { run: RunData | null }) {
  const claim = run?.dossier?.claims?.[0]
  return (
    <Block id="claim" label="The problem">
      <div className="grid items-end gap-10 md:grid-cols-5">
        <p className={`${lead} md:col-span-3`}>
          A result in a paper is a claim in words. Checking it costs days of broken code and missing hyperparameters.
        </p>
        <div className="md:col-span-2 md:pl-6">
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Kipf &amp; Welling report</div>
          <div className="mt-3 font-mono text-6xl font-light tracking-tight text-fg-3 sm:text-7xl">{claim ? showValue(claim.metric, claim.target) : ' '}</div>
        </div>
      </div>
    </Block>
  )
}

const STEPS = [
  { title: 'Read', text: 'Tavily finds the paper, the target value with a quote and the setup details, live. We keep only numbers that appear word for word in the text.' },
  { title: 'Rebuild', text: 'An agent reads the paper and the authors’ code as text, never run, writes its own version and runs it in isolated Nebius Sandboxes, one run per random seed.' },
  { title: 'Measure', text: 'An independent harness the agent cannot change loads the true labels itself, scores the predictions and compares them with the claim.' },
]

export function Steps() {
  return (
    <Block id="how" label="How Nullius works">
      <h2 className={lead}>Read. Rebuild. Measure.</h2>
      <ol className="m-0 mt-12 grid list-none gap-10 p-0 md:grid-cols-3 md:gap-12">
        {STEPS.map((s) => (
          <li key={s.title}>
            <h3 className="font-grotesk m-0 text-xl font-semibold tracking-[-0.03em] text-fg">{s.title}</h3>
            <p className="mb-0 mt-3 text-[15px] leading-relaxed text-fg-4">{s.text}</p>
          </li>
        ))}
      </ol>
    </Block>
  )
}

// A real finished check as a small certificate. The whole card opens it.
export function CertificatePreview({ run }: { run: RunData | null }) {
  const claim = run?.dossier?.claims?.[0]
  const v = run?.dossier?.verdicts?.[0]
  if (!claim || !v) return <Block id="certificate" label="A real certificate"><h2 className={lead}>Every check ends in a certificate.</h2><div className="mt-10 h-48 rounded-2xl border border-line" aria-hidden /></Block>
  const res = de2en(RESULT, v.urteil)
  const seeds = new Set<number>((run!.dossier.attempts ?? []).filter((a: { measured: number | null }) => a.measured != null).map((a: { seed: number }) => a.seed)).size
  const std = claim.metric === 'accuracy' ? `${(v.measured_std * 100).toFixed(1)}` : v.measured_std.toFixed(3)
  return (
    <Block id="certificate" label="A real certificate">
      <h2 className={lead}>Every check ends in a certificate: claimed against measured.</h2>
      <a href={EXAMPLE_RUN} className="group mt-10 block rounded-2xl border border-line bg-surface p-6 no-underline transition-colors hover:border-line-strong sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Badge tone={res.tone} title={resultHint(v.urteil)}>{res.label}</Badge>
          <span className="text-[12px] text-muted">Finished run, {run!.ref.title}</span>
        </div>
        <p className="font-grotesk mb-0 mt-5 max-w-xl text-lg font-semibold leading-snug tracking-[-0.02em] text-fg">{translateClaim(claim.text)}</p>
        <dl className="m-0 mt-8 grid grid-cols-2 gap-6">
          <div>
            <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Claimed</dt>
            <dd className="m-0 mt-2 font-mono text-4xl font-light tracking-tight text-fg-3 sm:text-5xl">{showValue(claim.metric, claim.target)}</dd>
          </div>
          <div>
            <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Measured</dt>
            <dd className="m-0 mt-2 font-mono text-4xl font-light tracking-tight text-tone-green sm:text-5xl">{showValue(claim.metric, v.measured_mean)}</dd>
            <div className="mt-1 font-mono text-[13px] text-fg-4">± {std} over {seeds} seeds</div>
          </div>
        </dl>
        <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4 text-[13px] text-fg-4">
          <span>Target value reached with our setup; method fidelity not checked.</span>
          <span className="inline-flex items-center gap-1.5 text-fg-3 group-hover:text-fg">Open the certificate <ArrowRight className="h-3.5 w-3.5" aria-hidden /></span>
        </div>
      </a>
    </Block>
  )
}

export function Proof() {
  const { paperOnly, withCode, placebo, reproducedAgree } = FIGURES
  const num = 'font-mono text-3xl font-light tracking-tight text-fg sm:text-4xl'
  return (
    <Block id="proof" label="Proof">
      <h2 className={lead}>When Nullius says a target was reached, it was.</h2>
      <div className="mt-10 grid gap-8 sm:grid-cols-2">
        <div>
          <div className={num}>{placebo.v4[0]} of {placebo.v4[1]}</div>
          <p className="mb-0 mt-2 text-[14px] leading-snug text-fg-4">deliberately wrong claims reported as reproduced (benchmark v4; also {placebo.v2[0]} of {placebo.v2[1]} in v2 and {placebo.v3[0]} of {placebo.v3[1]} in v3)</p>
        </div>
        <div>
          <div className={num}>{reproducedAgree.v4[0]} of {reproducedAgree.v4[1]}</div>
          <p className="mb-0 mt-2 text-[14px] leading-snug text-fg-4">&lsquo;reproduced&rsquo; verdicts in v4 agree with the reference (post hoc, small sample)</p>
        </div>
      </div>
      <p className={`${body} mt-10`}>
        How often it reaches the target depends on what the agent may read: <span className="font-mono text-fg">{paperOnly.matched} of {paperOnly.of}</span> units
        from the paper alone, <span className="font-mono text-fg">{withCode.matched} of {withCode.of}</span> with the authors&rsquo; code.
      </p>
      <div className="mt-8"><Matrix /></div>
      <p className="mb-0 mt-6 max-w-2xl text-[13px] leading-relaxed text-muted">
        A match means the target value was reached; method fidelity is not checked. <a href="#benchmark" className={link}>The full benchmark</a>
      </p>
    </Block>
  )
}

export function Lesson() {
  return (
    <Block id="lesson" label="Our own small bitter lesson">
      <h2 className={lead}>Days of cleverness left the count at {FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}. The authors&rsquo; code moved it.</h2>
      <div className="mt-8 flex max-w-2xl flex-col gap-4 text-[16px] leading-relaxed text-fg-4">
        <p className="m-0">
          We made the agent cleverer for days: better code extraction, repair loops, a fidelity checker. Agreement with the reference stayed at {FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}.
          Then the code model read the authors&rsquo; public code, as text, never run: {FIGURES.withCode.matched} of {FIGURES.withCode.of}. The missing details sat in their repository, not in a cleverer prompt.
        </p>
        <p className="m-0">
          Related to Rich Sutton&rsquo;s Bitter Lesson, not the same: we added information, not computation. In-sample, with one held-out miss ({HELD_OUT.matched} of {HELD_OUT.of}): an observation, not a proof.
        </p>
        <p className="m-0">
          <a href={`${REPO_BLOB}/${SOURCES.readme}`} {...ext} className={link}>Read the full lesson <ExternalLink className="h-3 w-3" aria-hidden /></a>
        </p>
      </div>
    </Block>
  )
}

const TAVILY_ROLES = [
  { icon: Search, title: 'Find papers', text: 'Discovery searches arXiv, GitHub and OpenReview for new papers on supported benchmarks.' },
  { icon: Quote, title: 'Read the claim', text: 'Extract reads the paper text. A target value counts only with a quote that appears word for word.' },
  { icon: FileText, title: 'Setup details', text: 'Hyperparameters and data splits from paper, README and chosen repository files, each with a quote.' },
  { icon: Bug, title: 'Error search', text: 'When generated code fails, Tavily searches the failing line and passes a hint to the repair.' },
  { icon: Users, title: 'Community check', text: 'After the verdict: issues and critiques. Findings that can be repeated run as sandbox experiments.' },
]

export function TavilySection() {
  return (
    <Block id="tavily" label="Tavily">
      <h2 className={lead}>Tavily is the researcher.</h2>
      <p className={`${body} mt-6`}>
        It answers questions the agent could not answer alone, live at run time. Every source is stored with a timestamp and a hash and shown on the certificate.
      </p>
      <ol className="m-0 mt-10 grid list-none gap-5 p-0">
        {TAVILY_ROLES.map((r) => (
          <li key={r.title} className="flex items-start gap-4">
            <r.icon className="mt-0.5 h-4 w-4 shrink-0 text-fg-3" aria-hidden />
            <p className="m-0 text-[15px] leading-relaxed text-fg-4"><span className="font-semibold text-fg">{r.title}.</span> {r.text}</p>
          </li>
        ))}
      </ol>
      <p className={`${body} mt-10`}>
        A finding that names something we can repeat runs in its own sandbox branch. In a live DLinear run, a seed-dependence claim ran as five extra seeds and was refuted.
      </p>
      <figure className="m-0 mt-6">
        <img src="img/findings-experiments-dark.jpg" alt="Findings that became experiments: three Tavily findings, each with its sandbox experiment and measurement" className="logo-dark w-full rounded-xl border border-line" loading="lazy" />
        <img src="img/findings-experiments-light.jpg" alt="" aria-hidden className="logo-light w-full rounded-xl border border-line" loading="lazy" />
        <figcaption className="mt-3 text-[12px] leading-relaxed text-muted">Words on the left, experiment in the middle, measurement on the right. Live DLinear run, 8 Oct 2026, paper-only mode.</figcaption>
      </figure>
      <details className="group mt-8 max-w-2xl">
        <summary className="inline-flex cursor-pointer list-none items-center gap-2 text-[14px] text-fg-3 hover:text-fg">
          Details and limits <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" aria-hidden />
        </summary>
        <ul className="m-0 mt-4 grid list-none gap-3 p-0 text-[14px] leading-relaxed text-fg-4">
          <li>A normal run calls Tavily live. The benchmark uses stored copies of Tavily results, because the key is scarce.</li>
          <li>Tavily&rsquo;s effect on the benchmark is small (2 vs 1 of 8 units). Its role is discovery, traceable sources and findings that become measurements.</li>
          <li>The community check never changes the verdict. Not every run finds something testable.</li>
          <li>The last-batch note for ETTh1 (<span className="font-mono">drop_last</span>) is not a Tavily find. It is a fixed list from our own research; the search did not return it.</li>
          <li>Not built: Tavily Research for citable sources, and possible explanations of a deviation as their own hypotheses.</li>
        </ul>
      </details>
      <p className="mb-0 mt-6 text-[14px]">
        <a href={README_TAVILY} {...ext} className={link}>How Nullius uses Tavily <ExternalLink className="h-3 w-3" aria-hidden /></a>
      </p>
    </Block>
  )
}

export function Limits() {
  const items = [...LIMITS, SCOPE_LIMIT]
  return (
    <Block id="limits" label="Limits">
      <details className="group">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-4">
          <h2 className={lead}>What this does not show</h2>
          <ChevronDown className="h-5 w-5 shrink-0 text-fg-4 transition-transform group-open:rotate-180" aria-hidden />
        </summary>
        <ul className="m-0 mt-8 grid list-none gap-x-10 gap-y-6 p-0 md:grid-cols-2">
          {items.map((l) => (
            <li key={l.title}>
              <h3 className="m-0 text-[14px] font-semibold text-fg">{l.title}</h3>
              <p className="mb-0 mt-1 text-[14px] leading-relaxed text-fg-4">{l.text}</p>
            </li>
          ))}
        </ul>
      </details>
    </Block>
  )
}

// The Discover sentence appears only when the live API answers and discovery is open; on the static mirror it is left out.
export function TryIt() {
  const { status } = useLiveStatus()
  const discovery = statusFor(status, 'discovery')
  return (
    <Block id="try" label="Try it">
      <h2 className={lead}>Check a claim yourself.</h2>
      <p className={`${body} mt-6`}>Pick a curated paper and start a quick check, or replay a finished one if the demo is busy.</p>
      <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-4">
        <a href="#catalog" className="inline-flex items-center gap-2 rounded-full bg-inverse px-6 py-3 text-sm font-medium text-on-inverse no-underline transition-colors hover:bg-inverse-hover">
          Open the catalog <ArrowRight className="h-4 w-4" aria-hidden />
        </a>
        {discovery?.live && <span className="text-[14px] text-fg-4">or <a href="#discover" className={link}>let Tavily find new papers</a></span>}
      </div>
    </Block>
  )
}

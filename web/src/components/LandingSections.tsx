import { useEffect, useState, type ReactNode } from 'react'
import { Bug, FileText, Quote, Search, Users } from 'lucide-react'
import { FIGURES, HELD_OUT, LIMITS, README_TAVILY, REPO_BLOB, SCOPE_LIMIT, SOURCES } from '../data/benchmark'
import { FEATURED_RUN_ID } from '../featured'
import { loadIndex, loadRun, type RunData } from '../data'
import { showValue } from '../format'
import { RESULT, de2en, resultHint, translateClaim } from '../labels'
import { useLiveStatus } from '../live'
import { statusFor } from '../livestatus'
import { Matrix } from './TwoModes'
import { Badge, Chevron, Disclosure, ExtLink, GoArrow, GoLink } from './ui'
import { Hint } from './Hint'

// The sections of the start page after the hero. They read as one story while scrolling: problem, how it works,
// one real certificate, Tavily, proof and lesson, limits, try it. One thought per section, at most one link.
// Every benchmark number is read from data/benchmark.ts; the certificate numbers come from the exported run.

export const EXAMPLE_RUN = `#/run/${FEATURED_RUN_ID}`
export const TIMELAPSE_RUN = `#/run/${FEATURED_RUN_ID}?play=1`

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
          A paper claims a number. Checking it costs days.
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
  { title: 'Read', text: 'Tavily finds the paper, the target and the setup, live. Every number comes with a quote.' },
  { title: 'Rebuild', text: 'An agent writes its own version from the paper and the authors’ code (read, never run) and runs it in Nebius Sandboxes.' },
  { title: 'Measure', text: 'An independent harness loads the true labels itself and scores the predictions.' },
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
      <h2 className={lead}>Every check ends in a certificate.</h2>
      <a href={EXAMPLE_RUN} className="nx-go group mt-10 block rounded-2xl border border-line bg-surface p-6 no-underline transition-[border-color,box-shadow,transform] duration-200 ease-out hover:border-line-strong motion-safe:hover:-translate-y-0.5 hover:shadow-md sm:p-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Badge tone={res.tone} title={resultHint(v.urteil)}>{res.label}</Badge>
          <span className="text-[12px] text-muted">Example run</span>
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
          <span>Method fidelity not checked.</span>
          <span className="inline-flex items-center gap-1.5 text-fg-3 group-hover:text-fg">Open <GoArrow /></span>
        </div>
      </a>
    </Block>
  )
}

export function Proof() {
  const { placebo, reproducedAgree } = FIGURES
  const num = 'font-mono text-3xl font-light tracking-tight text-fg sm:text-4xl'
  return (
    <Block id="proof" label="Proof">
      <h2 className={lead}>A reached target is a real one.</h2>
      <div className="mt-10 grid gap-8 sm:grid-cols-2">
        <div>
          <div className={num}>{placebo.v4[0]} of {placebo.v4[1]}</div>
          <p className="mb-0 mt-2 text-[14px] leading-snug text-fg-4">wrong claims reported as reproduced <Hint text={`Benchmark v4. The same check found ${placebo.v2[0]} of ${placebo.v2[1]} in v2 and in v3.`} /></p>
        </div>
        <div>
          <div className={num}>{reproducedAgree.v4[0]} of {reproducedAgree.v4[1]}</div>
          <p className="mb-0 mt-2 text-[14px] leading-snug text-fg-4">reproduced verdicts agree with the reference (post hoc)</p>
        </div>
      </div>
      <div className="mt-10"><Matrix short /></div>
      <p className="mb-0 mt-6 max-w-2xl text-[13px] leading-relaxed text-muted">
        Method fidelity not checked. <GoLink href="#benchmark">Full benchmark</GoLink>
      </p>
    </Block>
  )
}

export function Lesson() {
  return (
    <Block id="lesson" label="Our own small bitter lesson">
      <h2 className={lead}>Cleverness left {FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}. The authors&rsquo; code moved it.</h2>
      <p className={`${body} mt-6`}>
        With the authors&rsquo; code, read as text and never run: {FIGURES.withCode.matched} of {FIGURES.withCode.of}. Held-out: {HELD_OUT.matched} of {HELD_OUT.of}. An observation, not a proof.
      </p>
      <p className="mb-0 mt-4 text-[14px]">
        <ExtLink href={`${REPO_BLOB}/${SOURCES.readme}`}>The full lesson</ExtLink>
      </p>
    </Block>
  )
}

const TAVILY_ROLES = [
  { icon: Search, title: 'Find papers', text: 'on supported benchmarks' },
  { icon: Quote, title: 'Read the claim', text: 'LaTeX source (Tavily fallback)' },
  { icon: FileText, title: 'Setup details', text: 'each with a quote' },
  { icon: Bug, title: 'Error search', text: 'hints for the repair' },
  { icon: Users, title: 'Community check', text: 'after the verdict' },
]

export function TavilySection() {
  return (
    <Block id="tavily" label="Tavily">
      <h2 className={lead}>Tavily is the researcher.</h2>
      <p className={`${body} mt-6`}>Live, with sources stored and hashed.</p>
      <ol className="m-0 mt-8 grid list-none gap-3 p-0">
        {TAVILY_ROLES.map((r) => (
          <li key={r.title} className="flex items-start gap-4">
            <r.icon className="mt-0.5 h-4 w-4 shrink-0 text-fg-3" aria-hidden />
            <p className="m-0 text-[15px] leading-relaxed text-fg-4"><span className="font-semibold text-fg">{r.title}</span> {r.text}</p>
          </li>
        ))}
      </ol>
      <figure className="m-0 mt-8">
        <img src="img/findings-experiments-dark.jpg" alt="Findings that became experiments: three Tavily findings, each with its sandbox experiment and measurement" className="logo-dark w-full rounded-xl border border-line" loading="lazy" />
        <img src="img/findings-experiments-light.jpg" alt="" aria-hidden className="logo-light w-full rounded-xl border border-line" loading="lazy" />
        <figcaption className="mt-3 text-[12px] leading-relaxed text-muted">Finding, experiment, measurement. Live DLinear run.</figcaption>
      </figure>
      <Disclosure summary="Details and limits" className="mt-8 max-w-2xl">
        <ul className="m-0 mt-4 grid list-none gap-3 p-0 text-[14px] leading-relaxed text-fg-4">
          <li>A normal run calls Tavily live. The benchmark uses stored copies of Tavily results, because the key is scarce.</li>
          <li>Tavily&rsquo;s effect on the benchmark is small (2 vs 1 of 8 units). Its role is discovery, traceable sources and findings that become measurements.</li>
          <li>The community check never changes the verdict. Not every run finds something testable. In a live DLinear run, a seed-dependence claim ran as five extra seeds and was refuted.</li>
          <li>The last-batch note for ETTh1 (<span className="font-mono">drop_last</span>) is not a Tavily find. It is a fixed list from our own research; the search did not return it.</li>
          <li>Not built: Tavily Research for citable sources, and possible explanations of a deviation as their own hypotheses.</li>
        </ul>
      </Disclosure>
      <p className="mb-0 mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-[14px]">
        <GoLink href="#discover">Find new papers with Tavily</GoLink>
        <ExtLink href={README_TAVILY}>How Nullius uses Tavily</ExtLink>
      </p>
    </Block>
  )
}

export function Limits() {
  const items = [...LIMITS, SCOPE_LIMIT]
  return (
    <Block id="limits" label="Limits">
      <details>
        <summary className="nx-summary flex cursor-pointer items-center justify-between gap-4">
          <h2 className={lead}>What this does not show</h2>
          <Chevron className="h-5 w-5 text-fg-4" />
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
      <div className="mt-8 flex flex-wrap items-center gap-x-4 gap-y-4">
        <a href="#catalog" className="nx-go inline-flex items-center gap-2 rounded-full bg-inverse px-6 py-3 text-sm font-medium text-on-inverse no-underline transition-colors hover:bg-inverse-hover">
          Open the catalog <GoArrow className="h-4 w-4 shrink-0" />
        </a>
        {discovery?.live && <><span className="text-[14px] text-fg-4">or</span><GoLink href="#discover" className="text-[14px] text-fg underline underline-offset-4 hover:text-fg-max">let Tavily find new papers</GoLink></>}
      </div>
    </Block>
  )
}

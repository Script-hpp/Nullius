import { useEffect, useState, type ReactNode } from 'react'
import { Bug, Crop, FileText, GitCompare, Layers, Quote, Scale, ScrollText, Search, SearchCheck, Shuffle, Users, type LucideIcon } from 'lucide-react'
import { FIGURES, HELD_OUT, LIMITS, README_TAVILY, REPO_BLOB, SCOPE_LIMIT, SOURCES, UNITS } from '../data/benchmark'
import { useCatalog, evidenceHref } from '../catalog'
import { partA } from '../findings'
import { Compare, FINDING_ICONS } from './FindingsPage'
import { claimLine, juryEntries, splitFirstSentence } from '../landingBottom'
import { useInView } from '../useInView'
import { FEATURED_RUN_ID } from '../featured'
import { loadIndex, loadRun, type RunData } from '../data'
import { showValue } from '../format'
import { RESULT, de2en, resultHint, translateClaim } from '../labels'
import { useLiveStatus } from '../live'
import { statusFor } from '../livestatus'
import { Matrix } from './TwoModes'
import { Badge, Disclosure, ExtLink, GoArrow, GoLink } from './ui'
import { Hint } from './Hint'
import { measuredAttempts } from '../measurements'
import { formatSpread } from '../metricFormat'

// The sections of the start page after the hero. They read as one story while scrolling: problem, how it works,
// one real certificate, Tavily, proof and lesson, limits, try it. One thought per section, at most one link.
// Every benchmark number is read from data/benchmark.ts; the certificate numbers come from the exported run.

export const EXAMPLE_RUN = `#/run/${FEATURED_RUN_ID}`
export const TIMELAPSE_RUN = `#/run/${FEATURED_RUN_ID}?play=1`

const lead = 'font-grotesk m-0 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl'
const body = 'm-0 max-w-2xl text-[16px] leading-relaxed text-fg-4'

export function Block({ id, label, flush, children }: { id?: string; label: string; flush?: boolean; children: ReactNode }) {
  return (
    <section id={id} aria-label={label} className={`mx-auto max-w-4xl scroll-mt-8 px-6 ${flush ? 'pb-16 sm:pb-24' : 'py-16 sm:py-24'}`}>
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
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">{run?.ref.title ?? 'Paper'} reports</div>
          <div className="mt-3 font-mono text-6xl font-light tracking-tight text-fg-3 sm:text-7xl">{claim ? showValue(claim.metric, claim.target) : ' '}</div>
        </div>
      </div>
    </Block>
  )
}

const STEPS = [
  { title: 'Read', text: 'Read paper sources and quoted targets. Tavily researches repositories, setup details and repair hints at run time.' },
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
  const seeds = new Set<number>(measuredAttempts(run!.dossier, claim.id).map((a) => a.seed)).size
  const std = formatSpread(claim.metric, v.measured_std)
  return (
    <Block id="certificate" label="A real certificate">
      <h2 className={lead}>Every check ends in a certificate.</h2>
      <a href={EXAMPLE_RUN} className="nx-go group mt-10 block rounded-2xl border border-line bg-surface p-6 no-underline nx-card-lift sm:p-8">
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
          <span>Target value reached · method not verified.</span>
          <span className="inline-flex items-center gap-1.5 text-fg-3 group-hover:text-fg">Open <GoArrow /></span>
        </div>
      </a>
    </Block>
  )
}

export function ProofMatrix() {
  return (
    <Block label="Proof, the two modes" flush>
      <Matrix short />
      <p className="mb-0 mt-6 max-w-2xl text-[13px] leading-relaxed text-muted">
        Method fidelity not checked. <GoLink href="#benchmark">Full benchmark</GoLink>
      </p>
    </Block>
  )
}

export function Proof() {
  const { placebo, reproducedAgree } = FIGURES
  const num = 'font-mono text-3xl font-light tracking-tight text-fg sm:text-4xl'
  return (
    <Block id="proof" label="Proof">
      <h2 className={lead}>Independent measurements. Tested with false targets.</h2>
      <p className={`${body} mt-6`}>These controls describe benchmark v4, not a guarantee for other papers. Reaching a number does not verify the method.</p>
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
    </Block>
  )
}

// One row of unit squares: filled = the majority verdict matched the reference. The squares fill in once, left to right, when the row scrolls into view.
function UnitRow({ label, sub, key_, fill, matched, of }: { label: string; sub: string; key_: 'paperOnly' | 'withCode'; fill: string; matched: number; of: number }) {
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-x-6 gap-y-3 sm:grid-cols-[11rem_1fr_auto]">
      <div>
        <div className="text-[14px] font-semibold text-fg">{label}</div>
        <div className="text-[12px] text-muted">{sub}</div>
      </div>
      <ul className="col-span-2 row-start-2 m-0 flex list-none gap-1.5 p-0 sm:col-span-1 sm:row-start-auto sm:gap-2" aria-label={`${matched} of ${of} units matched the reference verdict`} style={{ '--nx-fill': fill } as React.CSSProperties}>
        {UNITS.map((u, i) => (
          <li key={u.id} title={`${u.author} ${u.model}: ${u[key_] ? 'matched' : 'missed'}`} className="min-w-0 flex-1">
            <div data-hit={u[key_]} className="nx-sq aspect-square w-full max-w-11 rounded-md border border-line-strong" style={{ '--nx-delay': `${i * 70}ms` } as React.CSSProperties} />
          </li>
        ))}
      </ul>
      <div className="col-start-2 row-start-1 font-mono text-2xl font-light tracking-tight text-fg sm:col-start-3 sm:text-right">{matched} of {of}</div>
    </div>
  )
}

export function Findings() {
  const [ref, seen] = useInView<HTMLUListElement>(0.1)
  return (
    <Block id="findings" label="What we found">
      <h2 className={lead}>The number in a paper is not enough to rebuild it.</h2>
      <p className={`${body} mt-6`}>
        Checking papers showed where the settings really live. Observations from our own checks, not a success rate.
      </p>
      <ul ref={ref} className={`${seen ? 'nx-in' : ''} m-0 mt-10 grid list-none gap-3 p-0 sm:grid-cols-2`}>
        {partA.slice(0, 4).map((f, i) => {
          const Icon = FINDING_ICONS[f.id] ?? ScrollText
          return (
            <li key={f.id} className="nx-reveal nx-card flex flex-col rounded-xl border border-line bg-surface p-5" style={{ '--nx-delay': `${i * 60}ms` } as React.CSSProperties}>
              <div className="flex items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-raised text-fg-3"><Icon className="h-4 w-4" aria-hidden /></span>
                <div className="min-w-0">
                  <div className="text-[12px] text-muted">{f.tag}</div>
                  <h3 className="m-0 text-[14px] font-semibold leading-snug text-fg">{f.headline}</h3>
                </div>
              </div>
              <Compare f={f} />
            </li>
          )
        })}
      </ul>
      <p className="mb-0 mt-8">
        <a href="#findings" className="inline-flex items-center gap-2 rounded-full border border-fg px-5 py-2.5 text-sm font-medium text-fg no-underline transition-colors hover:bg-fg hover:text-bg">All findings <GoArrow className="h-4 w-4 shrink-0" /></a>
      </p>
    </Block>
  )
}

export function Lesson() {
  const [ref, seen] = useInView<HTMLDivElement>(0.35)
  const { paperOnly, withCode } = FIGURES
  return (
    <Block id="lesson" label="Our own small bitter lesson">
      <h2 className={lead}>Cleverness left {paperOnly.matched} of {paperOnly.of}. The authors&rsquo; code moved it.</h2>
      <p className={`${body} mt-6`}>
        Same eight papers, two ways to rebuild them. The authors&rsquo; code is read as text and never run.
      </p>
      <div ref={ref} className={`${seen ? 'nx-in' : ''} mt-10 grid gap-8 rounded-2xl border border-line bg-surface p-5 sm:p-8`}>
        <UnitRow label="Paper only" sub="what the paper states" key_="paperOnly" fill="var(--c-fg-4)" matched={paperOnly.matched} of={paperOnly.of} />
        <UnitRow label="Paper and authors’ code" sub="plus the code, as text" key_="withCode" fill="var(--c-tone-green-solid)" matched={withCode.matched} of={withCode.of} />
        <div className="border-t border-line pt-6 grid grid-cols-[1fr_auto] items-center gap-x-6 gap-y-3 sm:grid-cols-[11rem_1fr_auto]">
          <div>
            <div className="text-[14px] font-semibold text-fg">Held out</div>
            <div className="text-[12px] text-muted">{HELD_OUT.author} {HELD_OUT.model}, not used to design the fixes</div>
          </div>
          <div className="col-span-2 row-start-2 flex items-center gap-3 sm:col-span-1 sm:row-start-auto">
            <div title={`${HELD_OUT.author} ${HELD_OUT.model}: missed`} className="aspect-square w-9 rounded-md border border-dashed border-tone-red" />
            <span className="text-[13px] text-fg-4">missed in both generations</span>
          </div>
          <div className="col-start-2 row-start-1 font-mono text-2xl font-light tracking-tight text-fg sm:col-start-3 sm:text-right">{HELD_OUT.matched} of {HELD_OUT.of}</div>
        </div>
      </div>
      <p className="mb-0 mt-4 max-w-2xl text-[13px] leading-relaxed text-muted">
        Filled square: the harness measurement matched the reference verdict. In-sample: the fixes were built after seeing these eight. An observation, not a proof.
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

const LIMIT_ICONS: Record<string, LucideIcon> = {
  'In-sample': Layers,
  'No "recall improved" claim': Scale,
  'Generation variance': Shuffle,
  'Different mode and models': GitCompare,
  'Method fidelity not checked': SearchCheck,
  [SCOPE_LIMIT.title]: Crop,
}

export function Limits() {
  const items = [...LIMITS, SCOPE_LIMIT]
  const [ref, seen] = useInView<HTMLUListElement>(0.1)
  return (
    <Block id="limits" label="Limits">
      <h2 className={lead}>Where the evidence stops.</h2>
      <p className={`${body} mt-6`}>{items.length} things these numbers do not show. Stated up front, so you can weigh them.</p>
      <ul ref={ref} className={`${seen ? 'nx-in' : ''} m-0 mt-10 grid list-none gap-3 p-0 sm:grid-cols-2 lg:grid-cols-3`}>
        {items.map((l, i) => {
          const Icon = LIMIT_ICONS[l.title] ?? Scale
          const { lead: first, rest } = splitFirstSentence(l.text)
          return (
            <li key={l.title} className="nx-reveal nx-card flex flex-col rounded-xl border border-line bg-surface p-5" style={{ '--nx-delay': `${i * 60}ms` } as React.CSSProperties}>
              <div className="flex items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-raised text-fg-3"><Icon className="h-4 w-4" aria-hidden /></span>
                <h3 className="m-0 text-[14px] font-semibold leading-snug text-fg">{l.title}</h3>
              </div>
              <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4">{first}</p>
              {rest && (
                <Disclosure summary="More" className="mt-2" summaryClassName="text-[13px] text-muted hover:text-fg">
                  <p className="mb-0 mt-2 text-[13px] leading-relaxed text-fg-4">{rest}</p>
                </Disclosure>
              )}
            </li>
          )
        })}
      </ul>
    </Block>
  )
}

// Closing call to action on an inverse band. Claim cards come from the exported catalog (jury-ready entries); if it fails to load the band still shows the button.
// The Discover sentence appears only when the live API answers and discovery is open; on the static mirror it is left out.
export function TryIt() {
  const { status } = useLiveStatus()
  const discovery = statusFor(status, 'discovery')
  const picks = juryEntries(useCatalog())
  const [ref, seen] = useInView<HTMLDivElement>(0.2)
  return (
    <Block id="try" label="Try it" flush>
      <div ref={ref} className={`${seen ? 'nx-in' : ''} rounded-3xl bg-inverse px-5 py-12 text-on-inverse sm:px-12 sm:py-16`}>
        <h2 className="font-grotesk m-0 max-w-3xl text-3xl font-semibold leading-snug tracking-[-0.03em] sm:text-4xl">Check a claim yourself.</h2>
        <p className="mb-0 mt-4 max-w-xl text-[16px] leading-relaxed opacity-75">Pick a claim. Nullius rebuilds the experiment and measures it. Recorded results open without sign-in.</p>
        {picks.length > 0 && (
          <ul className="m-0 mt-10 grid list-none gap-3 p-0 md:grid-cols-3">
            {picks.map((e, i) => {
              const href = evidenceHref(e) ?? '#catalog'
              return (
                <li key={e.id} className="nx-reveal" style={{ '--nx-delay': `${i * 80}ms` } as React.CSSProperties}>
                  <a href={href} className="nx-go group flex h-full flex-col justify-between gap-6 rounded-xl border border-on-inverse/20 p-5 text-on-inverse no-underline transition-colors hover:border-on-inverse/50 hover:bg-on-inverse/5">
                    <span className="font-grotesk text-[17px] font-semibold leading-snug tracking-[-0.02em]">{claimLine(e)}</span>
                    <span className="inline-flex items-center gap-1.5 text-[13px] opacity-75 group-hover:opacity-100">{e.run_id ? 'Open the result' : 'Open in catalog'} <GoArrow className="h-4 w-4 shrink-0" /></span>
                  </a>
                </li>
              )
            })}
          </ul>
        )}
        <div className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-4">
          <a href="#catalog" className="nx-go inline-flex items-center gap-2 rounded-full bg-on-inverse px-6 py-3 text-sm font-medium text-inverse no-underline transition-opacity hover:opacity-85">
            Open the catalog <GoArrow className="h-4 w-4 shrink-0" />
          </a>
          {discovery?.live && <><span className="text-[14px] opacity-75">or</span><GoLink href="#discover" className="text-[14px] text-on-inverse underline underline-offset-4">let Tavily find new papers</GoLink></>}
        </div>
      </div>
    </Block>
  )
}

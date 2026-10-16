import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Check, Bug, Crop, FileText, GitCompare, Layers, Quote, Ruler, Scale, ScrollText, Search, SearchCheck, Shuffle, Users, Wrench, BookOpenText, type LucideIcon } from 'lucide-react'
import { FIGURES, HELD_OUT, LIMITS, README_TAVILY, REPO_BLOB, SCOPE_LIMIT, SOURCES, UNITS } from '../data/benchmark'
import { heroCard, rampValue } from '../heroCard'
import { prefersReducedMotion } from './hero/config'
import { ScrollScene } from '../scroll/ScrollScene'
import { useCatalog, evidenceHref } from '../catalog'
import { partA } from '../findings'
import { Compare, FINDING_ICONS } from './FindingsPage'
import { claimLine, juryEntries, splitFirstSentence } from '../landingBottom'
import { useInView } from '../useInView'
import { FEATURED_RUN_ID } from '../featured'
import { loadIndex, loadRun, type RunData } from '../data'
import { RESULT, TARGET_HINT, de2en } from '../labels'
import { loadStageData, type StageData } from '../scroll/data'
import { useLiveStatus } from '../live'
import { statusFor } from '../livestatus'
import { Badge, Disclosure, ExtLink, GoArrow, GoLink } from './ui'
import { Hint } from './Hint'
import { measuredAttempts } from '../measurements'
import { formatMetric } from '../metricFormat'

// The sections of the start page after the hero. They read as one story while scrolling: problem, how it works,
// one real certificate, Tavily, proof and lesson, limits, try it. One thought per section, at most one link.
// Every benchmark number is read from data/benchmark.ts; the certificate numbers come from the exported run.

export const EXAMPLE_RUN = `#/run/${FEATURED_RUN_ID}`
export const TIMELAPSE_RUN = `#/run/${FEATURED_RUN_ID}?play=1`

const lead = 'font-grotesk m-0 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl'
const body = 'm-0 max-w-2xl text-[16px] leading-relaxed text-fg-4'

export function Block({ id, label, flush, inPage, children }: { id?: string; label: string; flush?: boolean; inPage?: boolean; children: ReactNode }) {
  // inPage: inside a page that already has its own column (benchmark page): no extra width cap or side padding
  return (
    <section id={id} aria-label={label} className={inPage ? 'scroll-mt-8 py-10 sm:py-14' : `mx-auto max-w-4xl scroll-mt-8 px-6 ${flush ? 'pb-14 sm:pb-20' : 'py-14 sm:py-20'}`}>
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

export function Problem() {
  return (
    <section id="claim" aria-label="The problem" className="mx-auto max-w-4xl scroll-mt-8 px-6 pb-6 pt-14 sm:pt-20">
      <h2 className={lead}>A paper claims a number. Checking it by hand takes days.</h2>
    </section>
  )
}

const STEPS: { title: string; icon: LucideIcon; text: string }[] = [
  { title: 'Read', icon: BookOpenText, text: 'Tavily finds and reads the sources: paper, code, setup details.' },
  { title: 'Rebuild', icon: Wrench, text: 'Nemotron on Nebius Token Factory writes the code. Nebius Sandboxes run it without internet.' },
  { title: 'Measure', icon: Ruler, text: 'Our own independent checker loads the true labels and scores the result.' },
]

const TAVILY_ROLES = [
  { icon: Search, title: 'Find papers', text: 'on supported benchmarks' },
  { icon: Quote, title: 'Read the claim', text: 'LaTeX source (Tavily fallback)' },
  { icon: FileText, title: 'Setup details', text: 'each with a quote' },
  { icon: Bug, title: 'Error search', text: 'hints for the repair' },
  { icon: Users, title: 'Community check', text: 'after the verdict' },
]

export function Steps() {
  return (
    <Block id="how" label="How Nullius works">
      <h2 className={lead}>Read. Rebuild. Measure.</h2>
      <ol className="m-0 mt-10 grid list-none gap-6 p-0 md:grid-cols-3 md:gap-8">
        {STEPS.map((s, i) => (
          <li key={s.title} className="flex flex-col rounded-xl border border-line bg-surface p-5 sm:p-6">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-raised text-fg-3"><s.icon className="h-[18px] w-[18px]" aria-hidden /></span>
              <h3 className="font-grotesk m-0 text-xl font-semibold tracking-[-0.03em] text-fg"><span className="mr-2 font-mono text-[13px] font-normal text-muted">{i + 1}</span>{s.title}</h3>
            </div>
            <p className="mb-0 mt-4 text-[15px] leading-relaxed text-fg-4">{s.text}</p>
          </li>
        ))}
      </ol>
      <Disclosure summary="How Tavily is used" className="mt-8 max-w-2xl">
        <ol className="m-0 mt-4 grid list-none gap-3 p-0">
          {TAVILY_ROLES.map((r) => (
            <li key={r.title} className="flex items-start gap-4">
              <r.icon className="mt-0.5 h-4 w-4 shrink-0 text-fg-3" aria-hidden />
              <p className="m-0 text-[15px] leading-relaxed text-fg-4"><span className="font-semibold text-fg">{r.title}</span> {r.text}</p>
            </li>
          ))}
        </ol>
        <ul className="m-0 mt-5 grid list-none gap-3 p-0 text-[14px] leading-relaxed text-fg-4">
          <li>A normal run calls Tavily live, with sources stored and hashed. The benchmark uses stored copies of Tavily results, because the key is scarce.</li>
          <li>Tavily&rsquo;s effect on the benchmark is small (2 vs 1 of 8 units). Its role is discovery, traceable sources and findings that become measurements.</li>
          <li>The community check never changes the verdict. Not every run finds something testable. In a live DLinear run, a seed-dependence claim ran as five extra seeds and was refuted.</li>
          <li>The last-batch note for ETTh1 (<span className="font-mono">drop_last</span>) is not a Tavily find. It is a fixed list from our own research; the search did not return it.</li>
          <li>Not built: Tavily Research for citable sources, and possible explanations of a deviation as their own hypotheses.</li>
        </ul>
        <p className="mb-0 mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-[14px]">
          <GoLink href="#discover">Find new papers with Tavily</GoLink>
          <ExtLink href={README_TAVILY}>How Nullius uses Tavily</ExtLink>
        </p>
      </Disclosure>
    </Block>
  )
}

// The result card of the hero: the featured run's own numbers. The measured value ramps up once when the card arrives (reduced motion: shown at once).
function useRamp(target: number | undefined, ms = 800): number | undefined {
  const [v, setV] = useState<number>(0)
  const raf = useRef(0)
  const still = prefersReducedMotion()
  useEffect(() => {
    if (target == null || still) return
    const t0 = performance.now()
    const tick = (now: number) => {
      setV(rampValue(target, now - t0, ms))
      if (now - t0 < ms) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [target, ms, still])
  return still ? target : target == null ? undefined : v
}

export function HeroCard({ run }: { run: RunData | null }) {
  const catalog = useCatalog()
  const claim = run?.dossier?.claims?.[0]
  const seeds = claim ? new Set<number>(measuredAttempts(run!.dossier, claim.id).map((a) => a.seed)).size : 0
  const card = heroCard(run, catalog?.find((e) => e.id === run?.ref.paper_id), seeds)
  const shown = useRamp(card?.measuredValue)
  if (!card || shown == null) return <div className="min-h-[19rem] rounded-2xl border border-line bg-surface" aria-hidden />
  const res = de2en(RESULT, card.urteil)
  return (
    <article className="nx-card-lift nx-rise relative rounded-2xl border border-line bg-surface p-6 text-left shadow-sm sm:p-8" aria-label="Example result">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Example check</span>
        <span className="inline-flex items-center gap-1.5"><Badge tone={res.tone}>{res.label}</Badge><span className="relative z-10"><Hint text={TARGET_HINT} /></span></span>
      </div>
      <p className="font-grotesk mb-0 mt-5 text-lg font-semibold leading-snug tracking-[-0.02em] text-fg sm:text-xl">{card.line} <span className="font-sans text-[13px] font-normal tracking-normal text-muted">· {card.hint}</span></p>
      <dl className="m-0 mt-7 grid grid-cols-2 gap-6">
        <div>
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Claimed</dt>
          <dd className="m-0 mt-2 font-mono text-4xl font-light tracking-tight text-fg-3 tabular-nums sm:text-5xl">{card.claimed}</dd>
        </div>
        <div>
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Measured</dt>
          <dd className="m-0 mt-2 font-mono text-4xl font-light tracking-tight text-tone-green tabular-nums sm:text-5xl" aria-label={card.measured}>{formatMetric(card.metric, shown)}</dd>
          {card.spread && <dd className="m-0 mt-1.5 font-mono text-[12px] text-muted">{card.spread} over seeds</dd>}
        </div>
      </dl>
      <p className="mb-0 mt-6 border-t border-line pt-4 text-[12px] leading-relaxed text-muted">{card.meta}</p>
      <a href={EXAMPLE_RUN} className="nx-go mt-3 inline-flex items-center gap-1.5 text-[13px] text-fg-3 no-underline after:absolute after:inset-0 after:content-[''] hover:text-fg">See the evidence <GoArrow /></a>
    </article>
  )
}

// The three plain numbers of "Does it work?", shared by the start page and the benchmark page. Small tests, never a success rate.
export function ProofNumbers() {
  const { placebo, withCode } = FIGURES
  const num = 'font-mono text-4xl font-light tracking-tight text-fg tabular-nums sm:text-5xl'
  const items = [
    { n: `${placebo.v4[0]} of ${placebo.v4[1]}`, t: 'false claims accepted', s: 'planted on purpose, as in the test above' },
    { n: `${withCode.matched} of ${withCode.of}`, t: 'known results matched', s: 'paper + authors’ code, in-sample' },
    { n: `${HELD_OUT.matched} of ${HELD_OUT.of}`, t: 'on a held-out paper', s: 'not used to design the fixes' },
  ]
  return (
    <ul className="m-0 grid list-none gap-8 p-0 sm:grid-cols-3" aria-label="Three small tests">
      {items.map((it) => (
        <li key={it.t} className="border-t border-line pt-5">
          <div className={num}>{it.n}</div>
          <p className="mb-0 mt-2 text-[15px] font-medium leading-snug text-fg">{it.t}</p>
          <p className="mb-0 mt-1 text-[13px] leading-snug text-muted">{it.s}</p>
        </li>
      ))}
    </ul>
  )
}

// Two rows that say what the test below is: both outcomes are the right ones. Numbers come from the same run the scene plays.
function TestLegend() {
  const [d, setD] = useState<StageData | null>(null)
  useEffect(() => { let on = true; loadStageData().then((x) => on && setD(x)).catch(() => {}); return () => { on = false } }, [])
  if (!d?.placebo || !d.placeboVerdict) return null
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`
  const rows = [
    { what: 'Real claim', n: pct(d.claim.target), out: d.verdict.urteil === 'reproduziert' ? 'Reproduced' : 'Checked', note: null },
    { what: 'Planted claim', n: pct(d.placebo.target), out: 'Rejected', note: 'correct' },
  ]
  return (
    <ul className="m-0 mt-5 grid max-w-md list-none gap-2 p-0" aria-label="What this test checks">
      {rows.map((r) => (
        <li key={r.what} className="flex items-center gap-3 rounded-lg border border-line bg-surface px-4 py-2.5 text-[15px] text-fg">
          <span className="w-28 shrink-0 text-fg-4">{r.what}</span>
          <span className="font-mono tabular-nums">{r.n}</span>
          <span aria-hidden className="text-subtle">&rarr;</span>
          <span className="font-medium">{r.out}</span>
          <Check className="h-4 w-4 shrink-0 text-tone-green" strokeWidth={3} aria-hidden />
          {r.note && <span className="text-[13px] font-medium text-tone-green">{r.note}</span>}
        </li>
      ))}
    </ul>
  )
}

export function DoesItWork() {
  return (
    <section id="proof" aria-label="Can it be fooled?" className="scroll-mt-8 pt-14 sm:pt-20">
      <div className="mx-auto max-w-4xl px-6">
        <h2 className={lead}>Can it be fooled?</h2>
        <p className={`${body} mt-3`}>In the same run we plant a false claim. Nullius has to reject it.</p>
        <TestLegend />
      </div>
      <div className="mt-6"><ScrollScene id="placebo" /></div>
      <div className="mx-auto max-w-4xl px-6 pt-6">
        <p className="mb-5 mt-0 text-[14px] text-fg-4">Three small tests. The first is the one above.</p>
        <ProofNumbers />
        <p className="mb-0 mt-8 text-[14px] text-fg-4">Small tests, not a success rate. <GoLink href="#benchmark" className="ml-1">Benchmark</GoLink></p>
      </div>
    </section>
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
      <p className={`${body} mt-4`}>Observations from our own checks, not a success rate.</p>
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

export function Lesson({ inPage = false }: { inPage?: boolean }) {
  const [ref, seen] = useInView<HTMLDivElement>(0.35)
  const { paperOnly, withCode } = FIGURES
  return (
    <Block id="lesson" label="Our own small bitter lesson" inPage={inPage}>
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

const LIMIT_ICONS: Record<string, LucideIcon> = {
  'In-sample': Layers,
  'No "recall improved" claim': Scale,
  'Generation variance': Shuffle,
  'Different mode and models': GitCompare,
  'Method fidelity not checked': SearchCheck,
  [SCOPE_LIMIT.title]: Crop,
}

const LIMIT_CHIPS = ['Two datasets: Cora and ETTh1, CPU only', 'Method fidelity not checked', 'Small, in-sample evaluation']

export function Limits() {
  const items = [...LIMITS, SCOPE_LIMIT]
  const [ref, seen] = useInView<HTMLUListElement>(0.1)
  return (
    <Block id="limits" label="Limits">
      <h2 className={lead}>Where the evidence stops.</h2>
      <ul className="m-0 mt-8 flex list-none flex-wrap gap-3 p-0">
        {LIMIT_CHIPS.map((c) => <li key={c} className="rounded-full border border-line-strong px-4 py-2 text-[14px] text-fg-3">{c}</li>)}
      </ul>
      <Disclosure summary="All limits" className="mt-8">
        <ul ref={ref} className={`${seen ? 'nx-in' : ''} m-0 mt-5 grid list-none gap-3 p-0 sm:grid-cols-2 lg:grid-cols-3`}>
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
                {rest && <p className="mb-0 mt-2 text-[13px] leading-relaxed text-fg-4">{rest}</p>}
              </li>
            )
          })}
        </ul>
      </Disclosure>
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

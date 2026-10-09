import type { ReactNode } from 'react'
import { ArrowRight, BookOpen, ExternalLink, FileCheck, GitBranch, LayoutGrid, Play, Search } from 'lucide-react'
import { FIGURES, HELD_OUT, LIMITS, README_TAVILY, REPO, REPO_BLOB, SCOPE_LIMIT, SOURCES, SUBMISSION_LINKS, SUTTON_URL } from '../data/benchmark'
import { FEATURED_RUN_ID } from '../featured'
import { Headline, Matrix, Precision } from './TwoModes'

// The sections of the start page after the hero. They follow one line, numbered in the kicker:
// problem, how it works, proof, the lesson, see for yourself, Tavily, limits, links. Every number is read from data/benchmark.ts.

export const EXAMPLE_RUN = `#/run/${FEATURED_RUN_ID}`
export const TIMELAPSE_RUN = `#/run/${FEATURED_RUN_ID}?play=1`

export function Kicker({ n, children }: { n: string; children: ReactNode }) {
  return (
    <h2 className="m-0 flex items-baseline gap-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">
      <span className="font-mono text-fg-4">{n}</span>{children}
    </h2>
  )
}

const ext = { target: '_blank', rel: 'noreferrer' } as const
const ul = 'text-fg underline underline-offset-4 hover:text-fg-max'

export function Proof() {
  const { paperOnly, withCode } = FIGURES
  return (
    <section id="proof" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-24 sm:pb-32">
      <Kicker n="03">Proof</Kicker>
      <p className="font-grotesk m-0 mt-5 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
        When Nullius says a target was reached, it was. It never confirmed a wrong claim.
      </p>
      <div className="mt-8"><Precision /></div>
      <p className="mb-0 mt-8 max-w-3xl text-[15px] leading-relaxed text-fg-4">
        How often it reaches the target depends on what the agent may read: <span className="font-mono text-fg">{paperOnly.matched} of {paperOnly.of}</span> units
        from the paper alone, <span className="font-mono text-fg">{withCode.matched} of {withCode.of}</span> with the authors&rsquo; code (in-sample; held-out {HELD_OUT.matched} of {HELD_OUT.of}).
        Two measurements, never added up. A match means the target value was reached; method fidelity is not checked.{' '}
        <a href="#benchmark" className={ul}>The full benchmark</a>
      </p>
    </section>
  )
}

export function Lesson() {
  return (
    <section id="lesson" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-24 sm:pb-32">
      <Kicker n="04">Our own small bitter lesson</Kicker>
      <p className="font-grotesk m-0 mt-5 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
        Days of cleverness left the count at {FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}. The authors&rsquo; code moved it.
      </p>
      <div className="mt-8 grid gap-10 lg:grid-cols-5">
        <div className="flex flex-col gap-4 text-[15px] leading-relaxed text-fg-4 lg:col-span-2">
          <p className="m-0">
            For days we made the agent cleverer: better code extraction, repair loops, a fidelity checker. Agreement with the reference verdicts stayed at{' '}
            <span className="font-mono text-fg">{FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}</span>.
          </p>
          <p className="m-0">
            Then we let the code model read the authors&rsquo; public code, as text, never run. Same units, same harness:{' '}
            <span className="font-mono text-fg">{FIGURES.withCode.matched} of {FIGURES.withCode.of}</span> matched. The missing details, such as a learning-rate schedule, sat in the authors&rsquo; repository, not in the paper and not in a cleverer prompt.
          </p>
          <p className="m-0">
            Rich Sutton&rsquo;s <a href={SUTTON_URL} {...ext} className={ul}>The Bitter Lesson</a> says computation beats hand-built knowledge. Ours is related, not the same: we did not add computation, we added <em>information</em>.
          </p>
          <p className="m-0 text-fg-3">
            In-sample, one held-out miss ({HELD_OUT.matched} of {HELD_OUT.of}): an observation, not a proof.{' '}
            <a href={`${REPO_BLOB}/${SOURCES.readme}`} {...ext} className={`inline-flex items-center gap-1 whitespace-nowrap ${ul}`}>Read the full lesson <ExternalLink className="h-3 w-3" aria-hidden /></a>
          </p>
        </div>
        <div className="min-w-0 space-y-6 lg:col-span-3">
          <Headline />
          <Matrix />
        </div>
      </div>
    </section>
  )
}

const LOOK: { icon: typeof FileCheck; title: string; text: string; href: string; ext?: boolean }[] = [
  { icon: FileCheck, title: 'Read a certificate', text: 'One finished check: claimed against measured, every seed, sources, branch tree.', href: EXAMPLE_RUN },
  { icon: Play, title: 'Watch it in 30 seconds', text: 'The same run replayed step by step.', href: TIMELAPSE_RUN },
  { icon: LayoutGrid, title: 'Open the catalog', text: 'Curated papers. Start a quick check, or replay a finished one if the demo is busy.', href: '#catalog' },
  { icon: Search, title: 'Discover new papers', text: 'Tavily searches for papers on benchmarks we can rebuild and proposes claims with a quote.', href: '#discover' },
]

export function SeeForYourself() {
  return (
    <section id="see" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-24 sm:pb-32">
      <Kicker n="05">See for yourself</Kicker>
      <ul className="m-0 mt-8 grid list-none gap-4 p-0 sm:grid-cols-2 lg:grid-cols-4">
        {LOOK.map((l) => (
          <li key={l.title}>
            <a href={l.href} className="group flex h-full flex-col rounded-xl border border-line bg-surface p-5 no-underline transition-colors hover:border-line-strong">
              <l.icon className="h-5 w-5 text-fg-4" aria-hidden />
              <h3 className="font-grotesk m-0 mt-4 text-lg font-semibold tracking-[-0.02em] text-fg">{l.title}</h3>
              <p className="mb-0 mt-2 flex-1 text-[13px] leading-relaxed text-fg-4">{l.text}</p>
              <span className="mt-4 inline-flex items-center gap-1.5 text-[12px] text-fg-3 group-hover:text-fg">Open <ArrowRight className="h-3.5 w-3.5" aria-hidden /></span>
            </a>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function TavilySection() {
  const roles = [
    'Discovery: finds new papers on supported benchmarks and proposes claims, each with a verbatim quote.',
    'Target value and setup details from the paper, README and repository, each with a quote.',
    'Error search while the agent repairs its code.',
    'Community check: what others found, with sources.',
  ]
  return (
    <section id="tavily" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-24 sm:pb-32">
      <Kicker n="06">Tavily</Kicker>
      <p className="font-grotesk m-0 mt-5 max-w-3xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
        Findings become experiments.
      </p>
      <div className="mt-8 grid gap-10 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <p className="m-0 text-[15px] leading-relaxed text-fg-4">
            A normal run calls Tavily live; every source is stored with a timestamp and a hash. Here Tavily does four jobs:
          </p>
          <ul className="m-0 mt-4 flex list-none flex-col gap-3 p-0 text-[14px] leading-relaxed text-fg-4">
            {roles.map((r) => <li key={r} className="flex gap-3"><GitBranch className="mt-1 h-3.5 w-3.5 shrink-0 text-fg-4" aria-hidden /><span>{r}</span></li>)}
          </ul>
          <p className="mb-0 mt-5 text-[15px] leading-relaxed text-fg-4">
            What it finds is not only reported. A finding that names something we can repeat runs in its own sandbox branch, and the harness measures it. In a live DLinear run, a seed-dependence claim ran as five extra seeds and was refuted, and a GitHub issue&rsquo;s MSE was compared with our own measurement, which supported the paper&rsquo;s value.
          </p>
          <p className="mb-0 mt-4 text-[13px] leading-relaxed text-muted">
            Not every run finds something testable, and this never changes the verdict. Tavily&rsquo;s effect on the benchmark is small; its role is discovery and traceable sources.{' '}
            <a href={README_TAVILY} {...ext} className={ul}>Details</a>
          </p>
        </div>
        <figure className="m-0 min-w-0 lg:col-span-3">
          <img src="img/findings-experiments-dark.jpg" alt="Findings that became experiments: three Tavily findings, each with its sandbox experiment and measurement" className="logo-dark w-full rounded-xl border border-line" loading="lazy" />
          <img src="img/findings-experiments-light.jpg" alt="" aria-hidden className="logo-light w-full rounded-xl border border-line" loading="lazy" />
          <figcaption className="mt-3 text-[12px] leading-relaxed text-muted">Words on the left, experiment in the middle, measurement on the right. A live DLinear run, 8 Oct 2026, paper-only mode.</figcaption>
        </figure>
      </div>
    </section>
  )
}

export function Limits() {
  const items = [...LIMITS, SCOPE_LIMIT]
  return (
    <section id="limits" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-24 sm:pb-32">
      <Kicker n="07">Honest limits</Kicker>
      <ul className="m-0 mt-8 grid list-none gap-x-10 gap-y-6 border-t border-line p-0 pt-8 md:grid-cols-2">
        {items.map((l) => (
          <li key={l.title}>
            <h3 className="m-0 text-[14px] font-semibold text-fg">{l.title}</h3>
            <p className="mb-0 mt-1 text-[14px] leading-relaxed text-fg-4">{l.text}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function Links() {
  return (
    <section id="links" className="mx-auto max-w-5xl scroll-mt-8 px-6 pb-28">
      <Kicker n="08">Links</Kicker>
      <ul className="m-0 mt-6 flex list-none flex-wrap gap-x-10 gap-y-3 border-t border-line p-0 pt-6 text-[15px]">
        <li><a href={REPO} {...ext} className={`inline-flex items-center gap-1.5 ${ul}`}>GitHub <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a></li>
        <li><a href={`${REPO_BLOB}/README.md`} {...ext} className={`inline-flex items-center gap-1.5 ${ul}`}><BookOpen className="h-3.5 w-3.5" aria-hidden />README</a></li>
        {SUBMISSION_LINKS.map((l) => (
          <li key={l.label}>
            {l.href
              ? <a href={l.href} {...ext} className={`inline-flex items-center gap-1.5 ${ul}`}>{l.label} <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a>
              : <span className="text-fg-4">{l.label}: <span className="font-mono text-[13px] text-muted">TBD</span></span>}
          </li>
        ))}
      </ul>
    </section>
  )
}

import { BadgeCheck, Calendar, FileSearch, GitCompare, Minimize2, Quote, Ruler, ScrollText, SlidersHorizontal, Split, Scissors, type LucideIcon } from 'lucide-react'
import { INTRO, PART_A, PART_B, partA, partB, type Finding, type Side, type Tone } from '../findings'
import { Disclosure, ExtLink, GoLink } from './ui'
import { NextStep } from './page'

export const FINDING_ICONS: Record<string, LucideIcon> = {
  fits: SlidersHorizontal,
  sparsetsf: GitCompare,
  rlinear: Ruler,
  tsmixer: Minimize2,
  saraf: Calendar,
  cutoff: Scissors,
  fewshot: Split,
  quote: Quote,
  placebo: BadgeCheck,
}

const BAR: Record<Tone, string> = {
  amber: 'bg-tone-amber-solid',
  red: 'bg-tone-red-solid',
  green: 'bg-tone-green-solid',
  gray: 'bg-line-stronger',
}

function Value({ s }: { s: Side }) {
  return (
    <div className="relative min-w-0 overflow-hidden rounded-lg bg-raised px-3 pb-3 pt-4">
      <span className={`absolute inset-x-0 top-0 h-1 ${BAR[s.tone]}`} aria-hidden />
      <div className="text-[11px] leading-snug text-muted">{s.label}</div>
      <div className="font-grotesk mt-1 break-words text-xl font-semibold leading-tight tracking-[-0.02em] text-fg">{s.value}</div>
    </div>
  )
}

export function Compare({ f }: { f: Finding }) {
  if (!f.left || !f.right) return null
  return (
    <div className="mt-4 grid grid-cols-2 gap-2" role="group" aria-label={`${f.left.label}: ${f.left.value}. ${f.right.label}: ${f.right.value}.`}>
      <Value s={f.left} />
      <Value s={f.right} />
    </div>
  )
}

function Head({ f }: { f: Finding }) {
  const Icon = FINDING_ICONS[f.id] ?? ScrollText
  return (
    <div className="flex items-center gap-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-raised text-fg-3"><Icon className="h-4 w-4" aria-hidden /></span>
      <div className="min-w-0">
        <div className="text-[12px] text-muted">{f.tag}</div>
        <h3 className="m-0 text-[15px] font-semibold leading-snug text-fg">{f.headline}</h3>
      </div>
    </div>
  )
}

function Details({ f }: { f: Finding }) {
  return (
    <Disclosure summary="Details" className="mt-4 border-t border-line pt-3">
      <dl className="m-0 mt-3 grid gap-3">
        {f.details.map((d) => (
          <div key={d.label}>
            <dt className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted">{d.label}</dt>
            <dd className="m-0 mt-1 text-[14px] leading-relaxed text-fg-4">{d.text}</dd>
          </div>
        ))}
      </dl>
      {f.commits.length > 0 && <p className="mb-0 mt-3 text-[12px] text-muted">Change in the code: {f.commits.map((c) => <code key={c} className="mr-1.5 font-mono">{c}</code>)}</p>}
      <p className="mb-0 mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[14px]">
        {f.links.map((l) => l.href.startsWith('http') ? <ExtLink key={l.href} href={l.href}>{l.label}</ExtLink> : <GoLink key={l.href} href={l.href}>{l.label}</GoLink>)}
      </p>
    </Disclosure>
  )
}

function Card({ f }: { f: Finding }) {
  return (
    <li id={f.id} className="flex scroll-mt-8 flex-col rounded-xl border border-line bg-surface p-5">
      <Head f={f} />
      <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4">{f.text}</p>
      <Compare f={f} />
      <Details f={f} />
    </li>
  )
}

function Row({ f }: { f: Finding }) {
  return (
    <li id={f.id} className="scroll-mt-8 rounded-xl border border-line bg-surface p-5">
      <Head f={f} />
      <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4">{f.text}</p>
      <Details f={f} />
    </li>
  )
}

const kicker = 'm-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted'

export function FindingsPage() {
  return (
    <main className="mx-auto max-w-4xl px-4 pb-16 md:px-8">
      <header className="pb-8 pt-8 md:pt-14">
        <div className={kicker}>Findings</div>
        <h1 className="font-grotesk m-0 mt-3 text-4xl font-semibold leading-[1.02] text-fg sm:text-5xl">{INTRO.title}</h1>
        <p className="font-grotesk m-0 mt-6 max-w-2xl text-xl font-semibold leading-snug tracking-[-0.02em] text-fg">{INTRO.summary}</p>
        <p className="mb-0 mt-3 flex max-w-2xl items-center gap-2 text-[14px] italic text-muted"><FileSearch className="h-4 w-4 shrink-0" aria-hidden />{INTRO.analogy}</p>
      </header>
      <section aria-label={PART_A.title}>
        <h2 className="font-grotesk m-0 text-2xl font-semibold tracking-[-0.03em] text-fg">{PART_A.title}</h2>
        <p className="mb-0 mt-2 text-[14px] text-muted">{PART_A.sub}</p>
        <ul className="m-0 mt-6 grid list-none gap-3 p-0 sm:grid-cols-2">{partA.map((f) => <Card key={f.id} f={f} />)}</ul>
      </section>
      <section className="mt-14" aria-label={PART_B.title}>
        <h2 className="font-grotesk m-0 text-2xl font-semibold tracking-[-0.03em] text-fg">{PART_B.title}</h2>
        <p className="mb-0 mt-2 text-[14px] text-muted">{PART_B.sub}</p>
        <ul className="m-0 mt-6 grid list-none gap-3 p-0 sm:grid-cols-2">{partB.map((f) => <Row key={f.id} f={f} />)}</ul>
      </section>
      <NextStep title="Check one yourself" text="Pick a paper in the catalog and watch the check run, or see how the checker does on known results." label="Open the catalog" href="#catalog" />
      <p className="mb-0 mt-6 text-[14px]"><GoLink href="#benchmark">See the Benchmark</GoLink></p>
    </main>
  )
}

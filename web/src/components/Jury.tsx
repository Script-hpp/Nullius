import { ArrowRight, BookOpen, ExternalLink, FileCheck, FlaskConical, LayoutGrid, Play, ShieldCheck } from 'lucide-react'
import { FIGURES, HELD_OUT, REPO, REPO_BLOB, SUBMISSION_LINKS } from '../data/benchmark'
import { EXAMPLE_RUN, TIMELAPSE_RUN } from './LandingSections'

// "For judges: start here": five places to click, one sentence each, then the three numbers and the links.
const STEPS: { icon: typeof FileCheck; title: string; text: string; href: string }[] = [
  { icon: FileCheck, title: 'Read one certificate', text: 'Kipf & Welling report 81.5% on Cora. See the claim next to the measurement over every seed, the sources, and the branch tree.', href: EXAMPLE_RUN },
  { icon: Play, title: 'Watch it in 30 seconds', text: 'The same check replayed step by step: Tavily finds, the agent writes, the sandbox runs, the harness measures.', href: TIMELAPSE_RUN },
  { icon: LayoutGrid, title: 'Open the catalog or start a live check', text: 'Curated papers that were run end to end. A quick check takes minutes; if the demo is busy it shows a replay and says so.', href: '#catalog' },
  { icon: ShieldCheck, title: 'Check the evidence', text: 'The benchmark: no wrong claim confirmed, then 2 of 8 against 7 of 8 units, with the limits next to the numbers.', href: '#benchmark' },
  { icon: FlaskConical, title: 'Read our own small bitter lesson', text: 'Why more information moved the count and more cleverness did not. An observation, not a proof.', href: '#/lesson' },
]

export function Jury() {
  const { paperOnly, withCode, placebo } = FIGURES
  const numbers = [
    { v: `${placebo.v4[0]} of ${placebo.v4[1]}`, t: 'wrong (placebo) claims confirmed in benchmark v4' },
    { v: `${paperOnly.matched} of ${paperOnly.of}`, t: 'units matched the reference verdict, paper only' },
    { v: `${withCode.matched} of ${withCode.of}`, t: `with the authors’ code, in-sample; held-out ${HELD_OUT.matched} of ${HELD_OUT.of}` },
  ]
  const a = 'inline-flex items-center gap-1.5 text-fg underline underline-offset-4 hover:text-fg-max'
  return (
    <main className="mx-auto max-w-4xl px-4 pb-16 md:px-8">
      <header className="pb-10 pt-10 md:pt-14">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">For judges</div>
        <h1 className="font-grotesk m-0 mt-3 text-4xl font-semibold leading-[1.02] tracking-[-0.03em] text-fg sm:text-5xl">Start here</h1>
        <p className="mb-0 mt-6 max-w-2xl text-lg leading-relaxed text-fg-4">
          Nullius takes a claim from a machine-learning paper, rebuilds the experiment in a Nebius sandbox and lets an independent harness measure it. Five clicks show the whole idea.
        </p>
      </header>

      <ol className="m-0 grid list-none gap-4 p-0">
        {STEPS.map((s, i) => (
          <li key={s.title}>
            <a href={s.href} className="group flex items-start gap-3 rounded-xl border border-line bg-surface p-4 no-underline transition-colors hover:border-line-strong sm:gap-5 sm:p-6">
              <span className="font-mono text-2xl font-light leading-none text-fg-4 sm:text-3xl">{String(i + 1).padStart(2, '0')}</span>
              <span className="min-w-0 flex-1">
                <span className="flex items-start gap-2"><s.icon className="mt-1.5 h-4 w-4 shrink-0 text-fg-4" aria-hidden /><span className="font-grotesk text-lg font-semibold tracking-[-0.02em] text-fg">{s.title}</span></span>
                <span className="mt-2 block text-[14px] leading-relaxed text-fg-4">{s.text}</span>
              </span>
              <ArrowRight className="mt-1 hidden h-4 sm:block w-4 shrink-0 text-fg-4 transition-transform group-hover:translate-x-0.5 group-hover:text-fg" aria-hidden />
            </a>
          </li>
        ))}
      </ol>

      <section aria-label="The numbers" className="mt-14">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">The numbers, in three lines</div>
        <ul className="m-0 mt-4 grid list-none gap-4 p-0 md:grid-cols-3">
          {numbers.map((n) => (
            <li key={n.t} className="rounded-xl border border-line bg-surface p-5">
              <div className="font-mono text-3xl font-light tracking-tight text-fg">{n.v}</div>
              <p className="mb-0 mt-2 text-[13px] leading-relaxed text-fg-4">{n.t}</p>
            </li>
          ))}
        </ul>
        <p className="mb-0 mt-4 text-[13px] leading-relaxed text-muted">Two separate measurements, never added up. A match means the target value was reached; method fidelity is not checked.</p>
      </section>

      <section aria-label="Links" className="mt-14 border-t border-line pt-8">
        <ul className="m-0 flex list-none flex-wrap gap-x-10 gap-y-3 p-0 text-[15px]">
          <li><a href={REPO} target="_blank" rel="noreferrer" className={a}>GitHub <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a></li>
          <li><a href={`${REPO_BLOB}/README.md`} target="_blank" rel="noreferrer" className={a}><BookOpen className="h-3.5 w-3.5" aria-hidden />README</a></li>
          {SUBMISSION_LINKS.map((l) => (
            <li key={l.label}>
              {l.href
                ? <a href={l.href} target="_blank" rel="noreferrer" className={a}>{l.label} <ExternalLink className="h-3.5 w-3.5" aria-hidden /></a>
                : <span className="text-fg-4">{l.label}: <span className="font-mono text-[13px] text-muted">TBD</span></span>}
            </li>
          ))}
        </ul>
      </section>
    </main>
  )
}

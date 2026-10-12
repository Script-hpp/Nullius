import { Check, Minus, ShieldCheck, BookOpen } from 'lucide-react'
import { Chevron, ExtLink } from './ui'
import { UNITS, HELD_OUT, FIGURES, MODES, LIMITS, REPO_BLOB, SOURCES } from '../data/benchmark'

const link = (path: string) => `${REPO_BLOB}/${path}`

export function BenchmarkSummary() {
  return <div className="rounded-xl border border-line bg-surface p-5" aria-label="How to read the benchmark">
    <h2 className="m-0 text-lg font-semibold text-fg">Three questions, three separate checks</h2>
    <ol className="mb-0 mt-4 space-y-3 pl-5 text-[14px] leading-relaxed text-fg-3">
      <li><strong className="text-fg">Did it agree on familiar examples?</strong> With the paper and authors’ code, {FIGURES.withCode.matched} of {FIGURES.withCode.of} reference verdicts matched. The fixes were developed using these examples, so this is a development result.</li>
      <li><strong className="text-fg">Did it work on a new paper?</strong> {HELD_OUT.model} was kept separate from development. {HELD_OUT.matched} of {HELD_OUT.of} matched; the check missed the known target in both runs.</li>
      <li><strong className="text-fg">Did it confirm deliberately wrong targets?</strong> None: {FIGURES.placebo.v4[0]} of {FIGURES.placebo.v4[1]} in the paper-only evaluation, and {FIGURES.placebo.cg[0]} of {FIGURES.placebo.cg[1]} in the separate authors’ code evaluation. These checks are not combined.</li>
    </ol>
    <p className="mb-0 mt-4 text-[13px] leading-relaxed text-muted">These small evaluations do not establish a success rate for new papers. A target match checks the number; it does not verify the method.</p>
  </div>
}

function Mark({ ok, label }: { ok: boolean; label: string }) {
  return ok
    ? <span role="img" aria-label={label} className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-tone-green-solid text-white sm:h-7 sm:w-7"><Check className="h-3.5 w-3.5 sm:h-4 sm:w-4" strokeWidth={3} aria-hidden /></span>
    : <span role="img" aria-label={label} className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-line-stronger text-subtle sm:h-7 sm:w-7"><Minus className="h-3.5 w-3.5 sm:h-4 sm:w-4" aria-hidden /></span>
}

// Two separate measurements. The headline numbers sit side by side and are never added up.
export function Headline() {
  const cols = [
    { m: MODES.paperOnly, big: `${FIGURES.paperOnly.matched} of ${FIGURES.paperOnly.of}`, note: `${FIGURES.paperOnly.runs} runs, pre-registered` },
    { m: MODES.withCode, big: `${FIGURES.withCode.matched} of ${FIGURES.withCode.of}`, note: `in-sample · held-out RLinear ${HELD_OUT.matched} of ${HELD_OUT.of}` },
  ]
  return (
    <div className="grid gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-2">
      {cols.map(({ m, big, note }) => (
        <div key={m.label} className="bg-surface p-6">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted">{m.label}</div>
          <div className="mt-4 font-mono text-5xl font-light tracking-tight text-fg">{big}</div>
          <div className="mt-2 text-[15px] font-medium leading-snug text-fg">units matched the reference verdict</div>
          <p className="mb-0 mt-3 text-[13px] leading-relaxed text-fg-4">{note}</p>
          <ExtLink href={link(m.source)} className="mt-3 text-[12px] text-fg-3 underline underline-offset-4 hover:text-fg">Source: {m.source.split('/').pop()}</ExtLink>
        </div>
      ))}
    </div>
  )
}

export function Precision() {
  const p = FIGURES.placebo
  const rows = [
    { v: `${p.v4[0]} of ${p.v4[1]}`, t: 'wrong claims reported as reproduced (v4)', s: `also ${p.v2[0]} of ${p.v2[1]} in v2 and v3 · pre-registered` },
    { v: `${p.cg[0]} of ${p.cg[1]}`, t: "wrong claims confirmed, authors'-code run", s: '8 rejected, 1 failed in setup' },
    { v: `${FIGURES.reproducedAgree.v4[0]} of ${FIGURES.reproducedAgree.v4[1]}`, t: 'reproduced verdicts agree with the reference', s: 'v4, post hoc, small sample' },
  ]
  return (
    <div>
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted"><ShieldCheck className="h-4 w-4" aria-hidden />Precision first</div>
      <ul className="m-0 mt-4 grid list-none gap-4 p-0 md:grid-cols-3" aria-label="Precision in three numbers">
        {rows.map((r) => (
          <li key={r.t} className="rounded-xl border border-line bg-surface p-5">
            <div className="font-mono text-3xl font-light tracking-tight text-fg">{r.v}</div>
            <div className="mt-2 text-[14px] font-medium leading-snug text-fg">{r.t}</div>
            <p className="mb-0 mt-2 text-[12px] leading-relaxed text-fg-4">{r.s}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function Matrix({ short = false }: { short?: boolean }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[300px] border-collapse text-left" aria-label="Same eight units in two modes">
        <caption className="sr-only">Matches with the reference verdict per unit, paper only versus paper and authors&rsquo; code</caption>
        <thead>
          <tr className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted">
            <th scope="col" className="px-3 py-3 font-semibold sm:px-5">Unit</th>
            <th scope="col" className="w-[4.5rem] px-1 py-3 text-center text-[10px] font-semibold leading-tight sm:w-auto sm:px-3 sm:text-[11px]">Paper only</th>
            <th scope="col" className="w-[4.5rem] px-1 py-3 text-center text-[10px] font-semibold leading-tight sm:w-auto sm:px-3 sm:text-[11px]">Paper + code</th>
          </tr>
        </thead>
        <tbody>
          {UNITS.map((u) => (
            <tr key={u.id} className="border-t border-line">
              <th scope="row" className="px-3 py-2.5 text-[14px] font-medium text-fg sm:px-5 sm:py-3">{u.author} <span className="block font-normal text-fg-4 sm:inline">{u.model}</span></th>
              <td className="px-1 py-2.5 text-center sm:px-3 sm:py-3"><Mark ok={u.paperOnly} label={`${u.author} ${u.model}, paper only: ${u.paperOnly ? 'matched' : 'missed'}`} /></td>
              <td className="px-1 py-2.5 text-center sm:px-3 sm:py-3"><Mark ok={u.withCode} label={`${u.author} ${u.model}, paper and authors' code: ${u.withCode ? 'matched' : 'missed'}`} /></td>
            </tr>
          ))}
          <tr className="border-t border-line-strong bg-surface-2">
            <th scope="row" className="px-3 py-3 text-[14px] font-semibold text-fg sm:px-5">Matches, in-sample</th>
            <td className="px-1 py-3 text-center font-mono text-base whitespace-nowrap text-fg sm:px-3 sm:text-lg">{FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}</td>
            <td className="px-1 py-3 text-center font-mono text-base whitespace-nowrap text-fg sm:px-3 sm:text-lg">{FIGURES.withCode.matched} of {FIGURES.withCode.of}</td>
          </tr>
          <tr className="border-t border-line">
            <th scope="row" className="px-3 py-2.5 text-[14px] font-medium text-fg sm:px-5 sm:py-3">{HELD_OUT.author} <span className="block font-normal text-fg-4 sm:inline">{HELD_OUT.model}</span> <span className="block text-[11px] font-semibold uppercase tracking-[0.12em] text-tone-amber-fg sm:ml-1 sm:inline">held-out</span></th>
            <td className="px-1 py-2.5 text-center text-[12px] text-muted sm:px-3">not in v4</td>
            <td className="px-1 py-2.5 text-center sm:px-3"><span className="inline-flex flex-col items-center gap-1 whitespace-nowrap sm:flex-row sm:gap-2"><Mark ok={false} label={`${HELD_OUT.model} held-out: missed`} /><span className="font-mono text-[13px] text-fg-3">{HELD_OUT.matched} of {HELD_OUT.of}</span></span></td>
          </tr>
        </tbody>
      </table>
      <p className="m-0 border-t border-line px-3 py-3 text-[12px] sm:px-5 leading-relaxed text-fg-4">
        {short
          ? 'Check: majority verdict equals the reference. Columns are never added up.'
          : `Check: the majority verdict equals the reference. Held-out measured ${HELD_OUT.measured[0]} and ${HELD_OUT.measured[1]} against ${HELD_OUT.target}. Columns are never added up.`}
      </p>
    </div>
  )
}

function Limits() {
  return (
    <details className="group rounded-xl border border-line bg-surface">
      <summary className="nx-summary flex cursor-pointer items-center justify-between gap-3 px-5 py-4 text-[14px] font-medium text-fg">
        Read with care: what this does not show
        <Chevron className="h-4 w-4 text-fg-4" />
      </summary>
      <ul className="m-0 grid list-none gap-5 border-t border-line p-5 md:grid-cols-2">
        {LIMITS.map((l) => (
          <li key={l.title}>
            <h3 className="m-0 text-[13px] font-semibold text-fg">{l.title}</h3>
            <p className="mb-0 mt-1 text-[13px] leading-relaxed text-fg-4">{l.text}</p>
          </li>
        ))}
      </ul>
    </details>
  )
}

export function TwoModes() {
  return (
    <section aria-label="Two modes, two measurements" className="mb-16 space-y-8">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Two modes, never added up</div>
        <p className="font-grotesk m-0 mt-4 max-w-4xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
          Two rebuild modes. Separate evaluations.
        </p>
      </div>
      <BenchmarkSummary />
      <Precision />
      <Headline />
      <Matrix />
      <Limits />
      <p className="m-0 flex max-w-3xl items-start gap-3 text-[14px] leading-relaxed text-fg-3">
        <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-fg-4" aria-hidden />
        <span>Cleverness left {FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}; the authors&rsquo; code moved it. <ExtLink href={link(SOURCES.readme)} className="whitespace-nowrap text-fg underline underline-offset-4 hover:text-fg-max">The lesson</ExtLink></span>
      </p>
    </section>
  )
}

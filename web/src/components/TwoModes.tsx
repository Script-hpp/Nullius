import { Check, Minus, ShieldCheck, BookOpen, ChevronDown, ExternalLink } from 'lucide-react'
import { UNITS, HELD_OUT, FIGURES, MODES, LIMITS, BITTER_LESSON, REPO_BLOB, SOURCES } from '../data/benchmark'

const link = (path: string) => `${REPO_BLOB}/${path}`

function Mark({ ok, label }: { ok: boolean; label: string }) {
  return ok
    ? <span role="img" aria-label={label} className="inline-flex h-6 w-6 items-center sm:h-7 sm:w-7 justify-center rounded-full bg-tone-green-solid text-white"><Check className="h-4 w-4" strokeWidth={3} aria-hidden /></span>
    : <span role="img" aria-label={label} className="inline-flex h-6 w-6 items-center sm:h-7 sm:w-7 justify-center rounded-full border border-line-stronger text-subtle"><Minus className="h-4 w-4" aria-hidden /></span>
}

// Two separate measurements. The headline numbers sit side by side and are never added up.
export function Headline() {
  const cols = [
    { m: MODES.paperOnly, big: `${FIGURES.paperOnly.matched} of ${FIGURES.paperOnly.of}`, note: `${FIGURES.paperOnly.runs} runs, pre-registered primary metric` },
    { m: MODES.withCode, big: `${FIGURES.withCode.matched} of ${FIGURES.withCode.of}`, note: `development units, in-sample · held-out RLinear ${HELD_OUT.matched} of ${HELD_OUT.of}` },
  ]
  return (
    <div className="grid gap-px overflow-hidden rounded-xl border border-line bg-line md:grid-cols-2">
      {cols.map(({ m, big, note }) => (
        <div key={m.label} className="bg-surface p-6">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted">{m.label}</div>
          <div className="mt-4 font-mono text-5xl font-light tracking-tight text-fg">{big}</div>
          <div className="mt-2 text-[15px] font-medium leading-snug text-fg">units matched the reference verdict</div>
          <p className="mb-0 mt-3 text-[13px] leading-relaxed text-fg-4">{note}</p>
          <a href={link(m.source)} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[12px] text-fg-3 underline underline-offset-4 hover:text-fg">Source: {m.source.split('/').pop()} <ExternalLink className="h-3 w-3" aria-hidden /></a>
        </div>
      ))}
    </div>
  )
}

export function Precision() {
  const p = FIGURES.placebo
  const rows = [
    { v: `${p.v4[0]} of ${p.v4[1]}`, t: 'deliberately wrong claims reported as reproduced in v4', s: `also ${p.v2[0]} of ${p.v2[1]} in v2 and ${p.v3[0]} of ${p.v3[1]} in v3 · pre-registered negative control` },
    { v: `${p.cg[0]} of ${p.cg[1]}`, t: "wrong claims confirmed in the authors'-code run", s: '8 rejected, 1 without a verdict (that generation failed in setup)' },
    { v: `${FIGURES.reproducedAgree.v4[0]} of ${FIGURES.reproducedAgree.v4[1]}`, t: "'reproduced' verdicts in v4 agree with the reference", s: 'post hoc, small sample' },
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

export function Matrix() {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[320px] border-collapse text-left" aria-label="Same eight units in two modes">
        <caption className="sr-only">Matches with the reference verdict per unit, paper only versus paper and authors&rsquo; code</caption>
        <thead>
          <tr className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted">
            <th scope="col" className="px-3 py-3 font-semibold sm:px-5">Unit</th>
            <th scope="col" className="px-1 py-3 text-center text-[10px] font-semibold sm:px-3 sm:text-[11px]">Paper only</th>
            <th scope="col" className="px-1 py-3 text-center text-[10px] font-semibold sm:px-3 sm:text-[11px]">Paper + code</th>
          </tr>
        </thead>
        <tbody>
          {UNITS.map((u) => (
            <tr key={u.id} className="border-t border-line">
              <th scope="row" className="px-3 py-3 text-[14px] font-medium text-fg sm:px-5">{u.author} <span className="font-normal text-fg-4">{u.model}</span></th>
              <td className="px-3 py-3 text-center"><Mark ok={u.paperOnly} label={`${u.author} ${u.model}, paper only: ${u.paperOnly ? 'matched' : 'missed'}`} /></td>
              <td className="px-3 py-3 text-center"><Mark ok={u.withCode} label={`${u.author} ${u.model}, paper and authors' code: ${u.withCode ? 'matched' : 'missed'}`} /></td>
            </tr>
          ))}
          <tr className="border-t border-line-strong bg-surface-2">
            <th scope="row" className="px-5 py-3 text-[14px] font-semibold text-fg">Development units</th>
            <td className="px-3 py-3 text-center font-mono text-lg whitespace-nowrap text-fg">{FIGURES.paperOnly.matched} of {FIGURES.paperOnly.of}</td>
            <td className="px-3 py-3 text-center font-mono text-lg whitespace-nowrap text-fg">{FIGURES.withCode.matched} of {FIGURES.withCode.of}</td>
          </tr>
          <tr className="border-t border-line">
            <th scope="row" className="px-5 py-3 text-[14px] font-medium text-fg">{HELD_OUT.author} <span className="font-normal text-fg-4">{HELD_OUT.model}</span> <span className="ml-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-tone-amber-fg">held-out</span></th>
            <td className="px-3 py-3 text-center text-[12px] text-muted">not in v4</td>
            <td className="px-3 py-3 text-center"><span className="inline-flex items-center gap-2 whitespace-nowrap"><Mark ok={false} label={`${HELD_OUT.model} held-out: missed`} /><span className="font-mono text-[13px] text-fg-3">{HELD_OUT.matched} of {HELD_OUT.of}</span></span></td>
          </tr>
        </tbody>
      </table>
      <p className="m-0 border-t border-line px-5 py-3 text-[12px] leading-relaxed text-fg-4">
        Check mark: the majority verdict over the generations equals the reference verdict. Held-out measured {HELD_OUT.measured[0]} and {HELD_OUT.measured[1]} against a target of {HELD_OUT.target}. Different modes and models; the two columns are never added up.
      </p>
    </div>
  )
}

function Limits() {
  return (
    <details className="group rounded-xl border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-4 text-[14px] font-medium text-fg">
        Read with care: what this does not show
        <ChevronDown className="h-4 w-4 text-fg-4 transition-transform group-open:rotate-180" aria-hidden />
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
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Two modes, two separate measurements</div>
        <p className="font-grotesk m-0 mt-4 max-w-4xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
          Nullius is a high-precision checker. How often it reaches the target depends on what the agent may read.
        </p>
      </div>
      <Precision />
      <Headline />
      <Matrix />
      <Limits />
      <p className="m-0 flex max-w-3xl items-start gap-3 text-[14px] leading-relaxed text-fg-3">
        <BookOpen className="mt-0.5 h-4 w-4 shrink-0 text-fg-4" aria-hidden />
        <span>{BITTER_LESSON} <a href={link(SOURCES.readme)} target="_blank" rel="noreferrer" className="whitespace-nowrap text-fg underline underline-offset-4">Our own small bitter lesson</a></span>
      </p>
    </section>
  )
}

import { AlertCircle, BookOpen, Search, ShieldCheck } from 'lucide-react'
import { Hint } from '../Hint'
import { ModeNote } from '../ModeNote'
import { toPercent } from '../ui'
import { tavilyChip } from '../../tavilysummary'
import { colorForTone, getDelta } from '../../format'
import { OVERALL, TONE_VAR, isPreliminary, resultHint, type Tone } from '../../labels'
import type { Event, Json } from '../../data'

const FIDELITY_HINT = 'The number was measured by an independent program. Whether our code follows the paper’s method is not verified. A result is a measurement with our setup, not a finding about the paper.'

// The one card at the top of a run: verdict, claim in one sentence, claimed against measured, the way it was rebuilt and why the number can be trusted.
// Everything longer lives in the tabs below.
export function VerdictCard({ dossier, events, claim, verdict, res, sentence, fmt, fmtSpread, paperUrl, fidelityUnchecked, gesamt, isCatalog, notes, claimText, onLimits, onTavily }: {
  dossier: Json; events: Event[]; claim: Json; verdict: Json | undefined; res: { label: string; tone: Tone }; sentence: string
  fmt: (x: number) => string; fmtSpread: (x: number) => string; paperUrl?: string; fidelityUnchecked: boolean
  gesamt?: string; isCatalog: boolean; notes: number; claimText: string; onLimits: (tab: 'limits') => void; onTavily: () => void
}) {
  const delta = verdict ? getDelta(verdict.measured_mean, claim.target, claim.metric) : null
  const tol = claim.metric === 'accuracy' ? toPercent(claim.tolerance) : Number(claim.tolerance).toFixed(3)
  const color = colorForTone(res.tone)
  const chip = tavilyChip(events, dossier?.sources)
  const overall = isCatalog && gesamt && !isPreliminary(gesamt) ? OVERALL[gesamt] ?? null : null
  return (
    <section aria-label="The verdict" className="border-t-2 pb-8 pt-6 sm:pb-10 sm:pt-8" style={{ borderTopColor: TONE_VAR[res.tone] }}>
      <div className="grid items-start gap-x-14 gap-y-8 lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="min-w-0">
          <h2 title={resultHint(verdict?.urteil)} className={`font-grotesk m-0 text-4xl font-semibold tracking-[-0.04em] sm:text-5xl ${color}`}>{res.label}</h2>
          <p className="mb-0 mt-4 max-w-2xl text-lg leading-snug text-fg-2">{sentence}</p>
          <p className="mb-0 mt-4 max-w-2xl text-[14px] leading-relaxed text-fg-4"><span className="text-muted">Claim · </span>{claimText}</p>
        </div>
        <dl className="m-0 grid grid-cols-2 gap-6 lg:border-l lg:border-line lg:pl-10">
          <div>
            <dt className="text-[10px] font-semibold uppercase tracking-widest text-muted">Claimed</dt>
            <dd className="m-0 mt-2 font-mono text-2xl text-fg-3">{fmt(claim.target)}</dd>
            <dd className="m-0 mt-1 font-mono text-[11px] text-muted">tolerance ±{tol}</dd>
          </div>
          <div>
            <dt className="text-[10px] font-semibold uppercase tracking-widest text-muted">Measured</dt>
            <dd className={`m-0 mt-2 font-mono text-3xl font-bold leading-none ${color}`}>{verdict ? fmt(verdict.measured_mean) : '–'}</dd>
            <dd className="m-0 mt-1.5 font-mono text-[11px] text-muted">
              {verdict?.measured_std != null && <>± {fmtSpread(verdict.measured_std)} over seeds</>}
              {delta && <span className={`block ${delta.color}`}>{delta.text}</span>}
            </dd>
          </div>
        </dl>
      </div>
      <ul className="m-0 mt-8 flex list-none flex-wrap items-center gap-x-6 gap-y-2 border-t border-line p-0 pt-5 text-[12.5px] text-fg-4">
        <li><ModeNote dossier={dossier} events={events} compact /></li>
        {fidelityUnchecked && (
          <li className="inline-flex items-center gap-1.5"><AlertCircle className="h-3.5 w-3.5 text-muted" aria-hidden />Method fidelity not checked<Hint text={FIDELITY_HINT} /></li>
        )}
        {overall && <li>{overall}</li>}
        {chip && (
          <li><button type="button" onClick={onTavily} data-testid="tavily-chip" title="Open Sources & Tavily"
            className="inline-flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-[12.5px] text-fg-3 underline decoration-line-strong underline-offset-4 hover:text-fg">
            <Search className="h-3.5 w-3.5 text-muted" aria-hidden />{chip}</button></li>
        )}
        {notes > 0 && (
          <li><button type="button" onClick={() => onLimits('limits')} className="inline-flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-[12.5px] text-fg-3 underline decoration-line-strong underline-offset-4 hover:text-fg">
            {notes} {notes === 1 ? 'note' : 'notes'} about this run</button></li>
        )}
        <li className="max-w-full">
          <ShieldCheck className="mr-1.5 inline h-3.5 w-3.5 align-[-2px] text-muted" aria-hidden />Scored by an independent harness the agent cannot change.{' '}
          <button type="button" onClick={() => onLimits('limits')} className="cursor-pointer border-0 bg-transparent p-0 text-[12.5px] text-fg-3 underline decoration-line-strong underline-offset-4 hover:text-fg">Why trust it</button>
        </li>
        {paperUrl && (
          <li><a className="inline-flex items-center gap-1.5 font-medium text-tone-blue no-underline hover:text-tone-blue-hover" href={paperUrl} target="_blank" rel="noreferrer"><BookOpen className="h-3.5 w-3.5" />Paper</a></li>
        )}
      </ul>
    </section>
  )
}

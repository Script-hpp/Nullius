import { useId, useState } from 'react'
import { ChevronDown, RotateCcw } from 'lucide-react'
import { ResultBand } from './ResultChart'
import { Badge } from './ui'
import { RESULT, TONE_VAR, de2en, resultHint } from '../labels'
import { fromUnits, sliderRange, toUnits, verdictForSlider, whatIfEligible } from '../whatif'
import type { Json } from '../data'

// "Try to fool Nullius": the same measurements against a claim the reader drags. Only the verdict changes (rule: whatif.ts, same as the
// harness). The real verdict above stays untouched; this card is a what-if view and runs no new check.
export function WhatIf({ dossier, claim, verdict, fmt, fmtSpread }: {
  dossier: Json; claim: Json; verdict: Json | undefined; fmt: (x: number) => string; fmtSpread: (x: number) => string
}) {
  const id = useId()
  const vals: number[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claim.id && typeof a.measured === 'number').map((a: Json) => a.measured)
  const eligible = whatIfEligible(verdict, vals.length, claim.tolerance)
  const mean: number = verdict?.measured_mean
  const tol: number = claim.tolerance
  const start = eligible ? toUnits(mean, claim.target, tol) : 0
  const [units, setUnits] = useState<number | null>(null)   // null = the paper's claim
  if (!eligible) return null
  const u = units ?? start
  const claimed = units === null ? claim.target : fromUnits(mean, units, tol)
  const { min, max } = sliderRange(start)
  const v = verdictForSlider(mean, claimed, tol)
  const res = de2en(RESULT, v)
  const distance = Math.abs(mean - claimed)
  const k = distance / tol
  const real = units === null
  const away = claim.metric === 'accuracy' ? `${(distance * 100).toFixed(1)} pp` : distance.toFixed(3)
  // fixed axis: it covers the whole slider range and every measurement, so the band moves against standing dots
  const dom: [number, number] = [Math.min(fromUnits(mean, min, tol), ...vals) - tol, Math.max(fromUnits(mean, max, tol), ...vals) + tol]
  const verdictWord: Record<string, string> = { reproduziert: 'reproduced', teilweise: 'partially reproduced', 'nicht reproduziert': 'not reproduced' }

  return (
    <details id="sec-whatif" className="group scroll-mt-16 min-w-0 border-y border-line" aria-label="Try to fool Nullius">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-4 [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="text-[15px] font-medium text-fg-2">Try to fool Nullius</span>
          <span className="ml-3 text-[12px] text-muted">Drag the claimed value and watch the verdict change</span>
        </span>
        <ChevronDown className="h-4 w-4 shrink-0 text-muted transition-transform group-open:rotate-180" aria-hidden />
      </summary>
      <p className="mb-0 mt-1 max-w-3xl text-[14px] leading-relaxed text-fg-4">
        Our measurements stay where they are; only the verdict changes. This is how placebo claims are rejected. What-if view: same measurements, different claim. This does not run a new check.
      </p>

      <div className="mt-6 grid items-start gap-10 pb-8 lg:grid-cols-3">
        <div className="min-w-0">
          <label htmlFor={id} className="block text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">What if the paper had claimed …?</label>
          <div className="font-grotesk mt-3 text-4xl font-semibold tracking-[-0.04em] text-fg"><span className="font-mono text-3xl font-light">{fmt(claimed)}</span></div>
          <input id={id} type="range" min={min} max={max} step={0.1} value={u} onChange={(e) => setUnits(Number(e.target.value))}
            aria-valuetext={`claimed ${fmt(claimed)}: ${verdictWord[v]}`}
            className="mt-5 w-full cursor-pointer accent-fg" style={{ accentColor: TONE_VAR[res.tone] }} />
          <div className="mt-1 flex justify-between font-mono text-[11px] text-muted"><span>{fmt(fromUnits(mean, min, tol))}</span><span>{fmt(fromUnits(mean, max, tol))}</span></div>
          <button type="button" onClick={() => setUnits(null)} disabled={real}
            className="mt-5 inline-flex cursor-pointer items-center gap-2 rounded-full border border-line-strong bg-transparent px-4 py-2 text-[12px] font-medium text-fg transition-colors hover:bg-raised disabled:cursor-default disabled:opacity-40">
            <RotateCcw className="h-3.5 w-3.5" />Reset to the paper&rsquo;s claim
          </button>
        </div>
        <div className="min-w-0 lg:col-span-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2" aria-live="polite">
            <Badge tone={res.tone} title={resultHint(v)}>{res.label}</Badge>
            <span className="text-[14px] leading-relaxed text-fg-3">
              {real ? 'The paper’s own claim. ' : ''}A claim of {fmt(claimed)} would be {verdictWord[v]}: our mean is {fmt(mean)}, {away} away, {k.toFixed(1)} × the tolerance of ±{fmt(tol)}.
            </span>
          </div>
          <div className="mt-6">
            <ResultBand dossier={dossier} claimId={claim.id} tone={res.tone} fmt={fmt} fmtSpread={fmtSpread} claimed={claimed} domain={dom} showKey={false} />
          </div>
        </div>
      </div>
    </details>
  )
}

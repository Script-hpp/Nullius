import type { Json } from '../data'
import { TONE_VAR, type Tone } from '../labels'
import { Term } from './Glossed'

const LANE_PX = [0, -10, 10, -20, 20]   // vertical offsets of the dot lanes

// "Claimed vs. measured" in one picture (the only chart on top of the page; the per-seed bars sit under Details): the claimed value with its tolerance band and every measured seed as a dot.
// Plain HTML positioning (no SVG text) so labels stay readable on a phone.
export function ResultBand({ dossier, claimId, tone, fmt, fmtSpread, claimed, domain, showKey = true }: {
  dossier: Json
  claimId: string
  tone: Tone
  fmt: (x: number) => string                   // value format of the metric (percent for accuracy, plain number for MSE/MAE)
  fmtSpread: (x: number) => string
  claimed?: number                             // what-if view: draw this claim instead of the real one (the measurements stay)
  domain?: [number, number]                    // fixed axis range, so the band moves against standing dots
  showKey?: boolean
}) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const verdict = (dossier.verdicts ?? []).find((v: Json) => v.claim_id === claimId)
  const vals: number[] = (dossier.attempts ?? [])
    .filter((a: Json) => a.claim_id === claimId && typeof a.measured === 'number')
    .map((a: Json) => a.measured)
  if (!claim) return null
  const target: number = claimed ?? claim.target
  const lo = target - claim.tolerance
  const hi = target + claim.tolerance
  const all = [...vals, lo, hi]
  const pad = (Math.max(...all) - Math.min(...all)) * 0.3 || 0.02
  const min = domain ? domain[0] : Math.min(...all) - pad
  const max = domain ? domain[1] : Math.max(...all) + pad
  const pos = (x: number) => `${((x - min) / (max - min)) * 100}%`
  // Dots that would overlap move to the next free lane above or below the axis (deterministic, by value), so every run stays visible.
  const lanes: { x: number; lane: number }[] = []
  const laneEnd: number[] = LANE_PX.map(() => -Infinity)
  for (const x of [...vals].sort((u, v) => u - v)) {
    const at = ((x - min) / (max - min)) * 100
    let lane = laneEnd.findIndex((end) => at - end >= 2.6)
    if (lane < 0) lane = laneEnd.indexOf(Math.min(...laneEnd))
    laneEnd[lane] = at
    lanes.push({ x, lane })
  }
  const mean = typeof verdict?.measured_mean === 'number' ? verdict.measured_mean : null
  const std = typeof verdict?.measured_std === 'number' ? verdict.measured_std : null
  const color = TONE_VAR[tone]

  return (
    <div>
      {vals.length > 0 ? (
        <>
        {showKey && <ul className="m-0 mb-8 list-none space-y-2 p-0 text-[14px] leading-relaxed text-fg-3">
          <li className="flex items-start gap-3"><span className="mt-1.5 inline-block h-3 w-3 shrink-0 rounded-full border border-fg-4 bg-fg-4/40" /><span>Each dot is one <Term term="seed">run with a different random start</Term>.</span></li>
          <li className="flex items-start gap-3"><span className="mt-1.5 inline-block h-3 w-6 shrink-0 rounded-sm" style={{ background: 'var(--chart-target)', opacity: 0.3 }} /><span>The band is the allowed range around the claim ({fmt(lo)} to {fmt(hi)}), the <Term term="tolerance">tolerance</Term>.</span></li>
          <li className="flex items-start gap-3"><span className="mt-1.5 inline-block h-4 w-0.5 shrink-0 rounded-full" style={{ background: color }} /><span>The solid line is our mean over all runs; the dashed line is the claim.</span></li>
        </ul>}
        <figure className="m-0 mt-2" aria-label={`Claimed ${fmt(target)}; measured ${vals.map(fmt).join(', ')}`}>
          <div className="relative h-24 mx-2">
            {/* tolerance band and claimed line */}
            <div className="absolute top-6 h-12 rounded-sm" style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})`, background: 'var(--chart-target)', opacity: 0.14 }} />
            <div className="absolute top-3 h-18 w-px border-l border-dashed" style={{ left: pos(target), borderColor: 'var(--chart-target)', height: '4.5rem' }} />
            <div className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-[10px] font-semibold uppercase tracking-wider" style={{ left: pos(target), color: 'var(--chart-target)' }}>
              Claimed {fmt(target)}
            </div>
            {/* axis */}
            <div className="absolute top-[3.0rem] left-0 right-0 h-px bg-line-strong" />
            {/* one dot per seed */}
            {lanes.map(({ x, lane }, i) => (
              <div key={i} title={`Run ${i + 1}: ${fmt(x)}`} className="absolute top-[3.0rem] w-3 h-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-fg-4 bg-fg-4/40"
                style={{ left: pos(x), marginTop: LANE_PX[lane] }} />
            ))}
            {/* mean */}
            {mean !== null && (
              <>
                <div className="absolute top-[2rem] w-0.5 h-8 -translate-x-1/2 rounded-full" style={{ left: pos(mean), background: color }} />
                <div className="absolute top-[4.6rem] -translate-x-1/2 whitespace-nowrap text-[11px] font-mono font-semibold" style={{ left: pos(mean), color }}>
                  Our mean {fmt(mean)}{std !== null ? ` ± ${fmtSpread(std)}` : ''}
                </div>
              </>
            )}
          </div>
        </figure>
        </>
      ) : (
        <p className="m-0 text-sm text-muted italic">No valid measurements to plot.</p>
      )}

    </div>
  )
}

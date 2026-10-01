import type { Json } from '../data'
import { TONE_VAR, type Tone } from '../labels'
import { Term } from './Glossed'
import { Section } from './ui'

const LANE_PX = [0, -10, 10, -20, 20]   // vertical offsets of the dot lanes

// "Claimed vs. measured" in one picture: the claimed value with its tolerance band and every measured seed as a dot.
// Plain HTML positioning (no SVG text) so labels stay readable on a phone.
export function ResultChart({ dossier, claimId, tone, verdictText, fmt, fmtSpread }: {
  dossier: Json
  claimId: string
  tone: Tone
  verdictText: string                          // one sentence built from the data by the caller
  fmt: (x: number) => string                   // value format of the metric (percent for accuracy, plain number for MSE/MAE)
  fmtSpread: (x: number) => string
}) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const verdict = (dossier.verdicts ?? []).find((v: Json) => v.claim_id === claimId)
  const vals: number[] = (dossier.attempts ?? [])
    .filter((a: Json) => a.claim_id === claimId && typeof a.measured === 'number')
    .map((a: Json) => a.measured)
  if (!claim) return null
  const fidelityUnchecked = !verdict?.methode || verdict.methode === 'nicht geprueft'

  const lo = claim.target - claim.tolerance
  const hi = claim.target + claim.tolerance
  const all = [...vals, lo, hi]
  const pad = (Math.max(...all) - Math.min(...all)) * 0.3 || 0.02
  const min = Math.min(...all) - pad
  const max = Math.max(...all) + pad
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
    <Section title="Claimed vs. measured" hint={vals.length ? `${vals.length} ${vals.length === 1 ? 'run' : 'runs'}` : undefined}>
      <p className="text-[15px] md:text-base text-fg-2 leading-relaxed font-medium max-w-3xl m-0">{verdictText}</p>

      {vals.length > 0 ? (
        <figure className="m-0 mt-8" aria-label={`Claimed ${fmt(claim.target)}; measured ${vals.map(fmt).join(', ')}`}>
          <div className="relative h-24 mx-2">
            {/* tolerance band and claimed line */}
            <div className="absolute top-6 h-12 rounded-sm" style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})`, background: 'var(--chart-target)', opacity: 0.14 }} />
            <div className="absolute top-3 h-18 w-px border-l border-dashed" style={{ left: pos(claim.target), borderColor: 'var(--chart-target)', height: '4.5rem' }} />
            <div className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-[10px] font-semibold uppercase tracking-wider" style={{ left: pos(claim.target), color: 'var(--chart-target)' }}>
              Claimed {fmt(claim.target)}
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
                  Mean {fmt(mean)}{std !== null ? ` ± ${fmtSpread(std)}` : ''}
                </div>
              </>
            )}
          </div>
          <figcaption className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-muted leading-relaxed">
            <span className="inline-flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 rounded-full border border-fg-4 bg-fg-4/40" /> one dot per <Term term="seed">random start</Term></span>
            <span className="inline-flex items-center gap-1.5"><span className="inline-block w-4 h-2.5 rounded-sm" style={{ background: 'var(--chart-target)', opacity: 0.3 }} /> allowed <Term term="tolerance">tolerance</Term> around the claim ({fmt(lo)} to {fmt(hi)})</span>
          </figcaption>
        </figure>
      ) : (
        <p className="mt-6 text-sm text-muted italic">No valid measurements to plot.</p>
      )}

      <p className="mt-6 pt-4 border-t border-line/50 text-[12px] text-muted leading-relaxed m-0">
        {fidelityUnchecked
          ? <><b className="text-fg-4">Method fidelity not checked.</b> The number was measured by an independent program; whether our code follows the paper&rsquo;s method is not verified.</>
          : 'Method fidelity was assessed; see the evidence tab for details.'}
      </p>
    </Section>
  )
}

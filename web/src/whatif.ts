// The what-if rule of the certificate page ("Try to fool Nullius"). It is the verdict rule of the harness applied to a different claim:
//   src/repro_agent/harness/verify.py  judge(): distance = |measured - target|; <= tolerance reproduced, <= 3 x tolerance partial, else not reproduced.
//   src/repro_agent/loop.py  _verdict(): measured = mean over the seeds, tolerance fixed before the run, the verdict looks only at the distance.
// "Smaller is better" for MSE changes only how a number is shown, never the rule. No new check is run: the measurements stay as they are.
// Plain erasable TypeScript so that `node --test` can run the unit test without a build step.

export type WhatIfVerdict = 'reproduziert' | 'teilweise' | 'nicht reproduziert'

// eps = 0 reproduces the harness exactly. The slider moves on a grid of tenths of the tolerance, where a tie lands on a boundary and
// floating point error could tip it over; the page passes eps = tolerance * 1e-9 so a claim exactly one tolerance away still counts as inside.
export function verdictFor(mean: number, claimed: number, tolerance: number, eps = 0): WhatIfVerdict {
  const distance = Math.abs(mean - claimed)
  if (distance <= tolerance + eps) return 'reproduziert'
  if (distance <= 3 * tolerance + eps) return 'teilweise'
  return 'nicht reproduziert'
}

export const SLIDER_EPS = 1e-9   // relative to the tolerance (see above)

export function verdictForSlider(mean: number, claimed: number, tolerance: number): WhatIfVerdict {
  return verdictFor(mean, claimed, tolerance, tolerance * SLIDER_EPS)
}

// Slider position as a multiple of the tolerance: -1 means "claimed one tolerance below our mean".
export const toUnits = (mean: number, claimed: number, tolerance: number): number => (claimed - mean) / tolerance
export const fromUnits = (mean: number, units: number, tolerance: number): number => mean + units * tolerance

// Range that always covers the three zones on both sides and the real claim (a placebo can sit far away).
export function sliderRange(start: number, span = 6): { min: number; max: number } {
  return { min: Math.min(-span, Math.floor(start) - 1), max: Math.max(span, Math.ceil(start) + 1) }
}

// The page shows the card only for a verdict that rests on measurements and on the distance.
export function whatIfEligible(verdict: { urteil?: string; measured_mean?: unknown } | undefined, nMeasured: number, tolerance: unknown): boolean {
  if (!verdict || typeof verdict.measured_mean !== 'number' || !Number.isFinite(verdict.measured_mean)) return false
  if (typeof tolerance !== 'number' || !(tolerance > 0) || nMeasured < 1) return false
  return verdict.urteil === 'reproduziert' || verdict.urteil === 'teilweise' || verdict.urteil === 'nicht reproduziert'
}

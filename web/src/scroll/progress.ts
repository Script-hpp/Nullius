// Pure scroll maths for the landing scenes (tested in progress.test.ts).

// Scroll distance of a scene in viewport heights: a short accent, never a long sticky stretch (2 at most, 1.3 at least).
export function scrollLengthVh(durationMs: number): number {
  return Math.min(2, Math.max(1.3, durationMs / 10500))
}

// 0..1 while the sticky stage is pinned: the section's top edge moves from 0 to -(height - viewport).
export function sceneProgress(top: number, height: number, viewport: number): number {
  const travel = height - viewport
  if (!(travel > 0)) return 0
  return Math.min(1, Math.max(0, -top / travel))
}

// A short hold at both ends: the first frame is seen before the film moves, and the last frame stays while the page is about to release.
const HOLD = 0.04

// Film time for a scroll progress. Reduced motion (and phones) show the poster frame and never scrub.
export function sceneTime(progress: number, duration: number, opts: { still: boolean; poster?: number }): number {
  if (opts.still) return Math.min(duration, Math.max(0, opts.poster ?? duration - 200))
  const p = Math.min(1, Math.max(0, (progress - HOLD) / (1 - 2 * HOLD)))
  return p * duration
}

// Light easing of the shown time towards the scroll target, independent of the frame rate. Snaps when it is within 1 ms.
export function easeToward(current: number, target: number, dtMs: number, tauMs = 110): number {
  const diff = target - current
  if (Math.abs(diff) < 1) return target
  return current + diff * (1 - Math.exp(-Math.max(0, dtMs) / tauMs))
}

// Pure play-clock maths for the landing scenes (tested in progress.test.ts).

// Share of the frame that must be on screen before a shot plays.
export const VISIBLE_AT = 0.5

export const isVisibleEnough = (ratio: number): boolean => ratio >= VISIBLE_AT

// Film time after `elapsed` ms of playing, from `start` to `duration`, then held on the last frame.
export function playTime(elapsed: number, start: number, duration: number): number {
  return Math.min(duration, Math.max(start, start + Math.max(0, elapsed)))
}

export const isFinished = (t: number, duration: number): boolean => t >= duration

// One rAF tick: the new film time after `dtMs` of real time. A paused clock stands still. dt is capped so a background tab or a
// long frame does not skip a stretch of the film.
export function advance(t: number, dtMs: number, duration: number, playing: boolean): number {
  if (!playing) return t
  return Math.min(duration, t + Math.min(100, Math.max(0, dtMs)))
}

// What the page shows for a scene that does not play: the poster frame (reduced motion, phones).
export function stillTime(duration: number, poster?: number): number {
  return Math.min(duration, Math.max(0, poster ?? duration - 200))
}

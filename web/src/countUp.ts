// Count-up for the run counters: value after `elapsed` ms of a `duration` ms ease-out (cubic) from 0 to `target`.
export const COUNT_MS = 700

export const easeOut = (t: number) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3)

export function countAt(target: number, elapsed: number, duration = COUNT_MS): number {
  if (!(duration > 0) || elapsed >= duration) return target
  return Math.round(target * easeOut(elapsed / duration))
}

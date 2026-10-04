import type { Event } from './data'

export type Timings = { seed0: number | null; total: number | null }
export type Eta = { n: number; seed0: number | null; total: number | null; totalMin: number | null; totalMax: number | null }

// Seconds from run_start to the first scored seed (first attempt with a measurement) and to run_end, from the event timestamps (seconds).
export function runTimings(events: Event[]): Timings {
  const start = events.find((e) => e.type === 'run_start')?.ts
  if (typeof start !== 'number') return { seed0: null, total: null }
  const first = events.find((e) => e.type === 'attempt' && e.measured != null)?.ts
  const end = events.find((e) => e.type === 'run_end')?.ts
  return { seed0: typeof first === 'number' ? first - start : null, total: typeof end === 'number' ? end - start : null }
}

const median = (xs: number[]) => {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

// Median over finished runs; only runs with a run_end count (a run without it never finished). Nothing is invented: no runs, no estimate.
export function etaFromRuns(runs: Event[][]): Eta | null {
  const t = runs.map(runTimings).filter((x) => x.total != null)
  if (!t.length) return null
  const totals = t.map((x) => x.total as number)
  return {
    n: t.length,
    seed0: median(t.flatMap((x) => (x.seed0 == null ? [] : [x.seed0]))),
    total: median(totals),
    totalMin: Math.min(...totals),
    totalMax: Math.max(...totals),
  }
}

export const minutes = (s: number) => (s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`)

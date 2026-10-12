import type { RunRef } from './data'

// The overview groups runs by the claim id inside each dossier. Until every dossier has settled (loaded or failed) the grouping
// would change several times, and cards would re-mount and re-animate. The page renders cards only once this is true.
export function allSettled(runs: RunRef[], settled: ReadonlySet<string>): boolean {
  return runs.every((r) => settled.has(r.id))
}

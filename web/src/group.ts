import type { RunRef } from './data'

export type RunGroup = { key: string; latest: RunRef; earlier: RunRef[] }

// One card per paper claim: the newest run is the card, older runs of the same claim are listed under it (no run is dropped).
// claimOf is the claim id from the loaded dossier; before it is loaded the paper alone is the key.
export function groupRuns(runs: RunRef[], claimOf: (r: RunRef) => string | undefined): RunGroup[] {
  const byKey = new Map<string, RunRef[]>()
  for (const r of runs) {
    const claim = claimOf(r)
    const key = claim ? `${r.paper_id}|${claim}` : r.paper_id
    byKey.set(key, [...(byKey.get(key) ?? []), r])
  }
  return [...byKey.entries()]
    .map(([key, rs]) => {
      const sorted = [...rs].sort((a, b) => b.timestamp.localeCompare(a.timestamp))
      return { key, latest: sorted[0], earlier: sorted.slice(1) }
    })
    .sort((a, b) => b.latest.timestamp.localeCompare(a.latest.timestamp))
}

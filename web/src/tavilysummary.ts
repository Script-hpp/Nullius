import type { Event } from './data'

// What the seven Tavily roles do, in plain words. One line over the "Sources & Tavily" tab counts the calls of this run by role.
export const TAVILY_ROLE: Record<number, string> = {
  1: 'find material', 2: 'read claims', 3: 'collect objections', 4: 'setup details', 5: 'search errors', 6: 'clarify deviations', 7: 'cite sources',
}

export function tavilySummary(events: Event[]): { total: number; parts: string[] } {
  const by = new Map<number, number>()
  let total = 0
  for (const e of events) {
    if (e.type !== 'tavily_call') continue
    total++
    const r = Number(e.role)
    if (Number.isFinite(r)) by.set(r, (by.get(r) ?? 0) + 1)
  }
  const parts = [...by.entries()].sort((a, b) => a[0] - b[0]).map(([r, n]) => `${TAVILY_ROLE[r] ?? `role ${r}`} ×${n}`)
  return { total, parts }
}

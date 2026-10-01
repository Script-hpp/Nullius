// "Play this check in 30 seconds": the events of a finished run, played back compressed. The total length is fixed, whatever the real duration.
import type { Event, Json } from './data'

export const TIMELAPSE_MS = 30_000

// Events of the shown claim plus the ones that belong to no claim (run start, sources, plan text).
export const eventsForClaim = (events: Event[], claimId: string): Event[] => events.filter((e) => e.claim_id === undefined || e.claim_id === claimId)

// How many events are visible after `elapsedMs`: spread evenly over the fixed length, many events per frame when the run had many.
export function countAt(elapsedMs: number, total: number, lengthMs = TIMELAPSE_MS): number {
  if (total <= 0) return 0
  if (elapsedMs >= lengthMs) return total
  return Math.min(total, Math.max(0, Math.floor((elapsedMs / lengthMs) * total) + 1))
}

// "This run took 41 minutes": from the first to the last event.
export function durationText(events: Event[]): string | null {
  const ts = events.map((e) => e.ts).filter((t) => typeof t === 'number' && Number.isFinite(t))
  if (ts.length < 2) return null
  const s = Math.max(...ts) - Math.min(...ts)
  if (s < 90) return `${Math.max(1, Math.round(s))} seconds`
  const min = Math.round(s / 60)
  if (min < 90) return `${min} minutes`
  const h = Math.floor(min / 60)
  return `${h} h ${min % 60} min`
}

// The dossier as it looked at that moment: the measurements that had come in, the verdict only once its event is there. At the end: the real dossier.
export function partialDossier(dossier: Json, claimId: string, seen: Event[], atEnd: boolean): Json {
  if (atEnd) return dossier
  const attempts = seen.filter((e) => e.type === 'attempt' && e.claim_id === claimId).map((e) => ({ claim_id: claimId, seed: e.seed, measured: e.measured ?? null, status: e.status, code_hash: e.code_hash }))
  const verdictSeen = seen.some((e) => e.type === 'verdict' && e.claim_id === claimId)
  return { ...dossier, attempts, verdicts: verdictSeen ? dossier.verdicts : [] }
}

// The closing lines of the time-lapse: exactly what the certificate says, taken from the same strings (label and verdict sentence are computed once
// in RunView and handed over; nothing is worded again here).
export function closingLines(o: { label: string; sentence: string; fidelityUnchecked: boolean; overall?: string | null }): { label: string; sentence: string; note: string | null } {
  return { label: o.label, sentence: o.sentence, note: o.fidelityUnchecked ? 'Method fidelity not checked.' : o.overall ?? null }
}

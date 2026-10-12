import type { Json } from './data'

// `chosen` is recorded by the orchestrator after candidate selection and repairs.
// Older runs have no selection markers and already contain their measured seeds.
export function measuredAttempts(dossier: Json, claimId: string): Json[] {
  const attempts: Json[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId)
  const selected = attempts.some((a) => typeof a.chosen === 'boolean')
  return attempts.filter((a) => (!selected || a.chosen === true) && typeof a.measured === 'number' && Number.isFinite(a.measured))
}

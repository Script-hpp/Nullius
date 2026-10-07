import type { Json } from './data'
import type { Tone } from './labels'

// Objection checks (Tavily role 3, stage 2): findings of the community search that were classified and, where measurable, tested in their own branch.
// Pure helpers; every number comes from `dossier.objection_checks`. Reported only, never part of the verdict.

export const TYPE_LABEL: Record<string, string> = {
  seed_variance: 'Seed dependence',
  parameter_change: 'Setting changed',
  reported_number: 'Reported number',
  not_measurable: 'Not measurable',
}

export const STATUS: Record<string, { label: string; tone: Tone }> = {
  bestaetigt: { label: 'Confirmed', tone: 'amber' },
  widerlegt: { label: 'Refuted', tone: 'green' },
  offen: { label: 'Open', tone: 'gray' },
  'nicht pruefbar': { label: 'Not measurable', tone: 'gray' },
}

const f = (x: unknown, d = 4) => (typeof x === 'number' && Number.isFinite(x) ? x.toFixed(d) : '–')

// "Objection checked: <quote> -> measured X -> confirmed/refuted/open" in one sentence of what was measured.
export function measuredLine(o: Json): string {
  const m: Json = o.measured ?? {}
  if (o.type === 'reported_number') {
    const sup: Record<string, string> = { paper: "supports the paper's value", reported: 'supports the reported value', both: 'supports both values', neither: 'supports neither value' }
    return m.mean === undefined ? 'no comparison possible' : `Reported ${f(o.reported_value, 3)}; our measurement ${f(m.mean)} ${sup[m.supports] ?? ''}`.trim()
  }
  if (o.type === 'seed_variance' && m.mean !== undefined) return `${m.n_extra} extra seeds: mean ${f(m.mean)}, spread over all seeds ${f(m.std_pool)} (main run ${f(m.main_mean)})`
  if (o.type === 'parameter_change' && m.delta !== undefined) {
    return `${paramName(o.param)} = ${o.to_value}: mean ${f(m.mean)} against ${f(m.main_mean)} for the main code (difference ${m.delta >= 0 ? '+' : ''}${f(m.delta)}, ${m.n} paired seeds)`
  }
  return 'no valid measurement'
}

export const paramName = (p: string | null | undefined) => (p ? p.replace(/_/g, ' ') : 'setting')

// Why an objection was not measured, in plain words (shown in the Evidence view).
export function whyNot(o: Json): string | null {
  if (o.type === 'not_measurable') return o.note ? `${capital(o.note)}.` : o.type_reason ? `${o.type_reason}` : 'A critique that cannot be turned into a measurement.'
  if (o.status === 'offen' && o.note) return `${capital(o.note)}.`
  return null
}
const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).replace(/\.$/, '')

export const checksFor = (dossier: Json, claimId: string): Json[] => ((dossier.objection_checks ?? []) as Json[]).filter((o) => !o.claim_id || o.claim_id === claimId)
export const branchCount = (o: Json) => ((o.attempts ?? []) as Json[]).length

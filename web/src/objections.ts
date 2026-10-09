import type { Json } from './data'
import type { Tone } from './labels'

// Objection checks (Tavily role 3, stage 2): findings of the community search that were classified and, where measurable, tested in their own branch.
// Pure helpers; every number comes from `dossier.objection_checks`. Reported only, never part of the verdict.

export const TYPE_LABEL: Record<string, string> = {
  seed_variance: 'Seed dependence',
  parameter_change: 'Setting changed',
  protocol_change: 'Evaluation protocol',
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
  if (o.type === 'protocol_change' && m.sizes) {
    const v = m.sizes[m.nearest_to_target]
    return `Same predictions, scored on the windows a drop_last loader keeps: largest change ${f(m.max_effect)} (batch size ${m.nearest_to_target}: ${f(v?.truncated_mean)} on ${v?.windows} of ${v ? v.windows + v.dropped : '–'} windows against ${f(v?.full_mean)} on all, ${m.n} seeds)`
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

// ---- "Words vs. measurement": one card per finding that became an experiment, plus the findings that did not ----

export type Point = { label: string; value: number; spread?: number; kind: 'ours' | 'reported' | 'main' }
export type Measure = { points: Point[]; target: number | null; tolerance: number | null }
export type ExperimentCard = {
  id: string
  type: string
  status: string
  quote: string
  url: string
  title: string          // what was run, one line
  details: string[]      // seeds, branches, rule
  outcome: string        // one sentence: what the measurement says about the words
  measure: Measure
}

const num = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
export const isTested = (o: Json) => o.type !== 'not_measurable' && o.status !== 'nicht pruefbar'

export function measureOf(o: Json, claim: Json | undefined): Measure {
  const m: Json = o.measured ?? {}
  const points: Point[] = []
  const add = (label: string, value: unknown, kind: Point['kind'], spread?: unknown) => { if (num(value)) points.push({ label, value, kind, spread: num(spread) ? spread : undefined }) }
  if (o.type === 'seed_variance') { add('main run', m.main_mean, 'main'); add('extra seeds', m.mean, 'ours', m.std_pool) }
  else if (o.type === 'reported_number') { add('our measurement', m.mean, 'ours'); add('reported in the issue', o.reported_value, 'reported') }
  else if (o.type === 'parameter_change') { add('main code', m.main_mean, 'main'); add(`${paramName(o.param)} = ${o.to_value}`, m.mean, 'ours') }
  else if (o.type === 'protocol_change' && m.sizes) {
    const v = m.sizes[m.nearest_to_target]
    add('all windows', v?.full_mean, 'main'); add('drop_last windows', v?.truncated_mean, 'ours')
  }
  return { points, target: num(claim?.target) ? claim.target : null, tolerance: num(claim?.tolerance) ? claim.tolerance : null }
}

export function experimentTitle(o: Json): string {
  const m: Json = o.measured ?? {}
  if (o.type === 'seed_variance') return `${m.n_extra ?? m.k ?? 'Extra'} extra seeds with the final code`
  if (o.type === 'parameter_change') return `${capital(paramName(o.param))} set to ${o.to_value}`
  if (o.type === 'protocol_change') return 'Same predictions, scored with a drop_last test loader'
  if (o.type === 'reported_number') return 'Reported number compared with our measurement'
  return TYPE_LABEL[o.type] ?? String(o.type)
}

export function experimentDetails(o: Json): string[] {
  const m: Json = o.measured ?? {}
  const out: string[] = []
  if (o.type === 'seed_variance' && Array.isArray(m.seeds)) out.push(`Seeds ${m.seeds.join(', ')} next to the ${m.n_main} main seeds`)
  if (o.type === 'reported_number') out.push('No new run: the number in the issue is checked against what the harness already measured')
  if (o.type === 'protocol_change' && m.sizes) out.push(`${m.n} seeds, batch sizes ${Object.keys(m.sizes).join(', ')}`)
  if (o.type === 'parameter_change' && m.n) out.push(`${m.n} paired seeds`)
  const b = branchCount(o)
  if (b > 0) out.push(`${b} sandbox branch${b === 1 ? '' : 'es'}: ${(o.attempts as Json[]).map((a) => a.branch_id).join(', ')}`)
  if (o.note) out.push(capital(String(o.note)) + '.')
  return out
}

export function outcomeLine(o: Json): string {
  const m: Json = o.measured ?? {}
  if (o.status === 'offen') return whyNot(o) ?? 'Not enough valid runs to decide'
  const hold = o.status === 'bestaetigt'
  if (o.type === 'seed_variance') return hold ? 'The result depends on the seed' : 'The result is stable across seeds'
  if (o.type === 'reported_number') return hold ? 'Our measurement matches the reported number' : m.supports === 'paper' ? "Our measurement supports the paper's value, not the reported one" : 'Our measurement does not match the reported number'
  if (o.type === 'parameter_change') return hold ? 'The setting changes the result' : 'The setting does not change the result'
  if (o.type === 'protocol_change') return hold ? 'The protocol changes the result beyond the tolerance' : 'The result stays within the tolerance under both protocols'
  return hold ? 'The objection holds in our setup' : 'The objection does not hold in our setup'
}

export function experimentCards(dossier: Json, claimId: string): ExperimentCard[] {
  const claim = ((dossier.claims ?? []) as Json[]).find((c) => c.id === claimId)
  return checksFor(dossier, claimId).filter(isTested).map((o) => ({
    id: String(o.id), type: o.type, status: o.status, quote: String(o.quote ?? ''), url: String(o.url ?? ''),
    title: experimentTitle(o), details: experimentDetails(o), outcome: outcomeLine(o), measure: measureOf(o, claim),
  }))
}

export type Discarded = { key: string; stage: 'dropped' | 'text'; reason: string; statement: string; quote: string; url: string }
const DROP_REASON: Record<string, string> = {
  quote_not_in_page: 'The quote is not in the page, so the statement cannot be verified',
  names_neither_method_title_nor_arxiv_id: 'The page names neither the method, the paper title nor the arXiv ID',
}

// Findings that never became a measurement: dropped by the quote check or the paper check, or kept as text because nothing can be measured.
export function discardedFindings(dossier: Json, claimId: string): Discarded[] {
  const out: Discarded[] = []
  const seen = new Set<string>()
  const add = (d: Discarded) => { if (!seen.has(d.key)) { seen.add(d.key); out.push(d) } }
  for (const it of ((dossier.community?.dropped_items ?? []) as Json[])) {
    if (!it.statement && !it.quote) continue   // page-level entries carry no statement
    add({ key: `${it.url}|${String(it.quote).slice(0, 60)}`, stage: 'dropped', reason: DROP_REASON[it.reason] ?? String(it.reason ?? 'Dropped').replace(/_/g, ' '), statement: String(it.statement ?? ''), quote: String(it.quote ?? ''), url: String(it.url ?? '') })
  }
  for (const o of checksFor(dossier, claimId).filter((x) => !isTested(x))) {
    add({ key: `${o.url}|${String(o.quote).slice(0, 60)}`, stage: 'text', reason: whyNot(o) ?? 'Nothing here can be measured', statement: String(o.statement ?? ''), quote: String(o.quote ?? ''), url: String(o.url ?? '') })
  }
  return out
}

// "n statements found · k kept · t tested · d dropped" from the stored raw candidates; falls back to the card counts when they are missing.
export function funnel(dossier: Json, claimId: string): { found: number; kept: number; tested: number; dropped: number } {
  const c: Json = dossier.community ?? {}
  const cands = (c.candidates ?? []) as Json[]
  const kept = ((c.findings ?? []) as Json[]).length
  const dropped = num(c.dropped) ? c.dropped : discardedFindings(dossier, claimId).filter((d) => d.stage === 'dropped').length
  return { found: cands.length || kept + dropped, kept, tested: experimentCards(dossier, claimId).length, dropped }
}

export function provenance(community: Json | null | undefined): string | null {
  if (!community) return null
  const d = community.queried_at ? new Date(community.queried_at) : null
  const at = d && !Number.isNaN(d.getTime()) ? d.toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC' : null
  const mode = community.mode === 'live' ? 'Live' : community.mode === 'cache' ? 'Saved result' : community.mode ? String(community.mode) : null
  return [mode, at].filter(Boolean).join(' · ') || null
}

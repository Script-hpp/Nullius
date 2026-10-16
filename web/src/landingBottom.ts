import type { CatalogEntry } from './catalog'
import { formatMetric } from './metricFormat.ts'

// Pure helpers for the lower part of the start page (Lesson, Limits, Try it).

// Catalog entries a visitor may start: jury-ready and with a stored result to open. Order of the catalog is kept.
export function juryEntries(catalog: CatalogEntry[] | null | undefined): CatalogEntry[] {
  return (catalog ?? []).filter((e) => e.jury?.ready && e.claim?.method && e.claim.target != null)
}

// "FITS reaches 0.372 MSE on ETTh1" built from the exported claim; parentheses in the method name ("GCN (this paper)") are dropped.
export function claimLine(e: CatalogEntry): string {
  const method = (e.claim.method ?? e.title).replace(/\s*\(.*?\)\s*/g, ' ').trim()
  const metric = e.claim.metric ?? 'accuracy'
  const value = metric === 'accuracy' ? formatMetric(metric, e.claim.target) : String(e.claim.target)
  const unit = metric === 'accuracy' ? 'accuracy' : metric.toUpperCase()
  return `${method} reaches ${value} ${unit}${e.dataset ? ` on ${e.dataset}` : ''}`
}

// First sentence of a limit as the card line, the rest (if any) behind a disclosure.
export function splitFirstSentence(text: string): { lead: string; rest: string } {
  const m = /^(.+?[.!?])\s+(?=[A-Z"'(])(.*)$/s.exec(text)
  return m ? { lead: m[1], rest: m[2] } : { lead: text, rest: '' }
}

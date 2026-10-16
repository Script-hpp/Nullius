import type { CatalogEntry } from './catalog'
import { formatMetric } from './metricFormat.ts'

// Pure text for a catalog card: the claim in plain words. Tested in catalogCard.test.ts.

const errorWord = (metric: string | undefined) => (metric === 'mae' ? 'forecast error (MAE)' : 'forecast error')

// "FITS: forecast error 0.372 on ETTh1", "GCN: accuracy 81.5% on Cora". Parentheses in the method name ("GCN (this paper)") are dropped.
export function cardClaim(e: Pick<CatalogEntry, 'claim' | 'dataset' | 'task' | 'title'>): string {
  const name = (e.claim.method ?? e.title.split(':')[0]).replace(/\s*\(.*?\)\s*/g, ' ').trim()
  const metric = e.claim.metric ?? 'accuracy'
  const where = e.dataset ?? e.task
  const value = metric === 'accuracy' ? formatMetric(metric, e.claim.target) : String(e.claim.target)
  const what = metric === 'accuracy' ? `accuracy ${value}` : `${errorWord(metric)} ${value}`
  return `${name}: ${what}${where ? ` on ${where}` : ''}`
}

// Plain label for an entry without a verdict.
export function noVerdictLabel(status: CatalogEntry['status'], hasReason: boolean): string {
  if (status === 'lauf_fehlgeschlagen') return 'Run failed'
  if (status === 'nicht_laufbar') return 'Cannot be rebuilt here'
  return hasReason ? 'Last check stopped' : 'Not checked yet'
}

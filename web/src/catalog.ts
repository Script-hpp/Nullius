import { useEffect, useState } from 'react'
import { getText, loadIndex } from './data'
import type { ReferenceBenchmark } from './referenceBench'
import { formatMetric, formatSpread } from './metricFormat'
import { DATA_CHANGED } from './live'

export type CatalogVerdict = {
  urteil: string
  measured_mean: number | null
  measured_std: number | null
  seeds: number
  target: number
  tolerance: number
  gesamtaussage: string
  graph_repairs?: number
  run_dir?: string
}

export type CatalogEntry = {
  id: string
  arxiv_id: string
  title: string
  origin?: 'discovery' | 'reference'   // missing in older exports = discovery
  benchmark?: ReferenceBenchmark | null
  dataset?: string
  task?: string
  claim: { method?: string; metric?: string; target?: number; horizon?: number; quote?: string; input_length?: number | null }
  protocol?: string
  column_check?: string
  github_url?: string | null
  source_url?: string | null
  status: 'ungeprueft' | 'geprueft' | 'lauf_fehlgeschlagen' | 'nicht_laufbar'
  reason?: string | null
  checked_at?: string | null
  cost_gross_usd?: number | null
  verdict: CatalogVerdict | null
  run_id: string | null
  jury: { ready: boolean; reason: string } | null
  // curated by catalog.yaml: a result that is shown with its date and a plain note, but not offered for a new check
  recorded?: { date: string; summary?: string; note: string } | null
  // compact paper head copied from the run's dossier.summary at export time (fallback when summaries.json has no entry)
  head?: { text?: string; authors?: string[]; authors_source?: string; year?: number | null; venue?: { name: string; quote: string }; institutions?: { name: string; quote: string }[]; institutions_source?: string; source?: string; generated_by?: string }
}

let cached: Promise<CatalogEntry[]> | null = null

// catalog.json is static and written by scripts/export_site_data.py; missing file = empty catalog.
export function loadCatalog(): Promise<CatalogEntry[]> {
  cached ??= Promise.all([getText('catalog.json', true), loadIndex()]).then(([text, runs]) => {
    const published = new Set(runs.map((r) => r.id))
    return (text ? JSON.parse(text) as CatalogEntry[] : []).map((entry) => ({
      ...entry,
      // The catalog can contain checks whose artifacts have not been published in this build.
      run_id: entry.run_id && published.has(entry.run_id) ? entry.run_id : null,
    }))
  })
  return cached
}
export const reloadCatalog = () => { cached = null; return loadCatalog() }

export function useCatalog(): CatalogEntry[] | null {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  useEffect(() => {
    let alive = true
    const load = () => loadCatalog().then((data) => { if (alive) setEntries(data) }).catch(() => { if (alive) setEntries([]) })
    const refresh = () => { cached = null; load() }
    load()
    window.addEventListener(DATA_CHANGED, refresh)
    return () => { alive = false; window.removeEventListener(DATA_CHANGED, refresh) }
  }, [])
  return entries
}

export const STATUS_TEXT: Record<CatalogEntry['status'], string> = {
  ungeprueft: 'Not run yet',
  geprueft: 'Checked',
  lauf_fehlgeschlagen: 'Run failed',
  nicht_laufbar: 'Not runnable',
}

const isShare = (metric?: string) => metric === 'accuracy'

// Accuracy is a share (shown as percent); MSE and MAE stay plain decimals.
export function fmtMeasured(e: CatalogEntry): string {
  const v = e.verdict
  if (!v || v.measured_mean == null) return '–'
  const std = v.measured_std == null ? '' : ` ± ${formatSpread(e.claim.metric, v.measured_std)}`
  return `${formatMetric(e.claim.metric, v.measured_mean)}${std}`
}

export function fmtClaimed(e: CatalogEntry): string {
  const t = e.claim.target
  if (t == null) return '–'
  return isShare(e.claim.metric) ? `${(t <= 1 ? t * 100 : t).toFixed(2)}%` : formatMetric(e.claim.metric, t)
}

// Measured minus claimed: percentage points for accuracy, absolute for MSE/MAE.
export function fmtDelta(e: CatalogEntry): { text: string; sign: -1 | 0 | 1 } | null {
  const v = e.verdict
  if (!v || v.measured_mean == null || v.target == null) return null
  const share = isShare(e.claim.metric)
  const diff = (v.measured_mean - v.target) * (share ? 100 : 1)
  const d = share ? 1 : 3
  if (Math.abs(diff) < 0.5 * 10 ** -d) return { text: `±0.${'0'.repeat(d)}${share ? ' pp' : ''}`, sign: 0 }
  return { text: `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(d)}${share ? ' pp' : ''}`, sign: diff > 0 ? 1 : -1 }
}

export const evidenceHref = (e: CatalogEntry, view?: 'log' | 'replay') => (e.run_id ? `#/run/${e.run_id}${view === 'replay' ? '?play=1' : view === 'log' ? '?tab=log' : ''}` : null)


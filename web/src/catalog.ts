import { useEffect, useState } from 'react'
import { getText } from './data'

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
}

let cached: Promise<CatalogEntry[]> | null = null

// catalog.json is static and written by scripts/export_site_data.py; missing file = empty catalog.
export function loadCatalog(): Promise<CatalogEntry[]> {
  cached ??= getText('catalog.json').then((t) => (t ? (JSON.parse(t) as CatalogEntry[]) : []))
  return cached
}

export function useCatalog(): CatalogEntry[] | null {
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null)
  useEffect(() => { loadCatalog().then(setEntries).catch(() => setEntries([])) }, [])
  return entries
}

export const STATUS_TEXT: Record<CatalogEntry['status'], string> = {
  ungeprueft: 'Not run yet',
  geprueft: 'Ran through the loop',
  lauf_fehlgeschlagen: 'Run failed',
  nicht_laufbar: 'Not runnable',
}

const isShare = (metric?: string) => metric === 'accuracy'

// Accuracy is a share (shown as percent); MSE and MAE stay plain decimals.
export function fmtMeasured(e: CatalogEntry): string {
  const v = e.verdict
  if (!v || v.measured_mean == null) return '–'
  const share = isShare(e.claim.metric)
  const k = share ? 100 : 1
  const d = share ? 1 : 3
  const std = v.measured_std == null ? '' : ` ± ${(v.measured_std * k).toFixed(d)}`
  return `${(v.measured_mean * k).toFixed(d)}${share ? '%' : ''}${std}`
}

export function fmtClaimed(e: CatalogEntry): string {
  const t = e.claim.target
  if (t == null) return '–'
  return isShare(e.claim.metric) ? `${(t <= 1 ? t * 100 : t).toFixed(2)}%` : t.toFixed(3)
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

export const evidenceHref = (e: CatalogEntry, tab?: 'log') => (e.run_id ? `#/run/${e.run_id}${tab ? `?tab=${tab}` : ''}` : null)

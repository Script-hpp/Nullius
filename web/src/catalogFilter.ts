import type { CatalogEntry } from './catalog'

export type CatalogFilter = 'featured' | 'all' | 'failed' | 'unchecked'
export function catalogFilterFromHash(hash: string): CatalogFilter {
  const view = new URLSearchParams(hash.split('?')[1] ?? '').get('view')
  return view === 'all' || view === 'failed' || view === 'unchecked' ? view : 'featured'
}

export function matchesCatalogFilter(entry: CatalogEntry, filter: CatalogFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'featured') return !!(entry.jury?.ready && entry.run_id)
  if (filter === 'unchecked') return entry.status === 'ungeprueft'
  return entry.status === 'lauf_fehlgeschlagen' || entry.status === 'nicht_laufbar' ||
    !!(entry.verdict && (entry.verdict.gesamtaussage === 'nicht bewertbar (Methode nicht umgesetzt)' ||
      !['reproduziert', 'teilweise'].includes(entry.verdict.urteil)))
}

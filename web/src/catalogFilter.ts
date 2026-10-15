import type { CatalogEntry } from './catalog'

export type CatalogFilter = 'featured' | 'all' | 'failed' | 'unchecked'
export function catalogFilterFromHash(hash: string): CatalogFilter {
  const view = new URLSearchParams(hash.split('?')[1] ?? '').get('view')
  return view === 'all' || view === 'failed' || view === 'unchecked' ? view : 'featured'
}

export function matchesCatalogFilter(entry: CatalogEntry, filter: CatalogFilter): boolean {
  if (filter === 'all') return true
  if (filter === 'featured') return !!((entry.jury?.ready || entry.recorded) && entry.run_id)   // recorded results are shown, but only jury-ready entries get a start button
  if (filter === 'unchecked') return entry.status === 'ungeprueft'
  return entry.status === 'lauf_fehlgeschlagen' || entry.status === 'nicht_laufbar' ||
    !!(entry.verdict && (entry.verdict.gesamtaussage === 'nicht bewertbar (Methode nicht umgesetzt)' ||
      !['reproduziert', 'teilweise'].includes(entry.verdict.urteil)))
}

// "9 Oct 2026" from an ISO date, independent of the viewer's time zone.
export function recordedDate(date: string): string {
  const d = new Date(`${date}T00:00:00Z`)
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

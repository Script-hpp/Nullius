// Paper head: summary, authors and affiliations of the paper behind a run. Data: public/data/discovery/summaries.json (scripts/make_summaries.py).
import type { PaperSummary } from './summaries'

// A run is found by its paper id (kipf_gcn_2017, disc_2511_08094) or by the arXiv id.
export function findHead(all: Record<string, PaperSummary>, key: string | undefined | null): (PaperSummary & { arxiv_id: string }) | undefined {
  if (!key) return undefined
  const m = key.match(/^disc_(\d{4})_(\d{4,5})$/)   // discovery runs are named after their arXiv id
  if (m && all[`${m[1]}.${m[2]}`]) return { ...all[`${m[1]}.${m[2]}`], arxiv_id: `${m[1]}.${m[2]}` }
  for (const [arxivId, e] of Object.entries(all)) if (arxivId === key || e.paper_ids?.includes(key)) return { ...e, arxiv_id: arxivId }
  return undefined
}

const SKIP = new Set(['of', 'the', 'and', 'for', 'de', 'du', 'des', 'van', 'von', 'la', 'le', 'di', 'da', 'der', 'at', 'in'])

// Monogram from the name only (no logo): initials of the words without small linking words, at most four; a one-word name gives its first three letters.
export function monogram(name: string): string {
  const words = name.replace(/\([^)]*\)/g, ' ').replace(/[–—-]/g, ' ').split(/\s+/).map((w) => w.replace(/[^\p{L}\p{N}]/gu, '')).filter((w) => w && !SKIP.has(w.toLowerCase()))
  if (words.length === 0) return '?'
  if (words.length === 1) return words[0].slice(0, 3).replace(/^./, (c) => c.toUpperCase())
  return words.slice(0, 4).map((w) => w[0].toUpperCase()).join('')
}

// "Kipf, Welling" / "A, B, C, D, E and 3 more": at most `max` names, the rest as a count.
export function authorLine(authors: string[], max = 5): { shown: string[]; more: number } {
  return { shown: authors.slice(0, max), more: Math.max(0, authors.length - max) }
}

// Venue wins over the arXiv year (the arXiv date is the first upload, the venue year is the publication).
export function whenLine(head: Pick<PaperSummary, 'year' | 'venue'>): string | null {
  return head.venue?.name ?? (head.year ? String(head.year) : null)
}

// First sentence of the summary, for the compact variant.
export function firstSentence(text: string): string {
  const m = text.match(/^.+?[.!?](?=\s|$)/)
  return (m ? m[0] : text).trim()
}

// The head a run wrote itself (dossier.summary, or the paper_summary event while it is live): same shape as an entry of summaries.json.
// It wins over summaries.json: it is what this run read from this paper.
export function headFromSummary(s: Record<string, any> | null | undefined, arxivId = ''): (PaperSummary & { arxiv_id: string }) | undefined {
  if (!s || (!s.text && !s.authors?.length && !s.institutions?.length)) return undefined
  return {
    summary: s.text ?? '', source: s.source ?? '', generated_by: s.generated_by, authors: s.authors, authors_source: s.authors_source, year: s.year,
    venue: s.venue, institutions: s.institutions, institutions_source: s.institutions_source, arxiv_id: arxivId,
  }
}

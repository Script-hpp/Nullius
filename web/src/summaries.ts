import { useEffect, useState } from 'react'

// Plain-language summary of a paper for readers without ML background.
// Stored per paper (public/data/discovery/summaries.json), generated once, never per page view.
export type PaperSummary = {
  summary: string            // two to three simple sentences
  source: string             // what the summary is based on
  generated_by?: string      // model role that wrote it
  example?: boolean          // true: placeholder text written by hand, not model output
  // Paper head (scripts/make_summaries.py): who wrote it, where, when. Every field has its own source.
  title?: string
  paper_ids?: string[]       // runs and catalog entries that belong to this arXiv paper
  authors?: string[]
  authors_source?: string
  year?: number | null
  venue?: { name: string; quote: string }
  institutions?: { name: string; quote: string }[]
  institutions_source?: string
}

let cache: Promise<Record<string, PaperSummary>> | null = null

export function loadSummaries(): Promise<Record<string, PaperSummary>> {
  cache ??= fetch(import.meta.env.BASE_URL + 'data/discovery/summaries.json')
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}))
  return cache
}

export function useSummary(arxivId: string): PaperSummary | undefined {
  const [s, setS] = useState<PaperSummary | undefined>()
  useEffect(() => {
    let alive = true
    loadSummaries().then((all) => { if (alive) setS(all[arxivId]) })
    return () => { alive = false }
  }, [arxivId])
  return s
}

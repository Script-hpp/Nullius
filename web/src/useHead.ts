import { useEffect, useState } from 'react'
import { findHead } from './paperhead'
import { loadSummaries, type PaperSummary } from './summaries'

export function useHead(key: string | undefined | null) {
  const [head, setHead] = useState<(PaperSummary & { arxiv_id: string }) | undefined>()
  useEffect(() => {
    let alive = true
    loadSummaries().then((all) => { if (alive) setHead(findHead(all, key)) })
    return () => { alive = false }
  }, [key])
  return head
}

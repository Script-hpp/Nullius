import { Search } from 'lucide-react'

export type TavilyKind = 'live' | 'stored'

// The one small Tavily brand: search icon plus the word, and "live" or "stored" when known. Live = fetched during the run, stored = saved copy.
export function TavilyMark({ kind, className = '', title }: { kind?: TavilyKind; className?: string; title?: string }) {
  const tip = title ?? (kind === 'live' ? 'Tavily was called live during this run.' : kind === 'stored' ? 'Tavily result stored earlier and loaded again; no new call.' : 'Live research by Tavily')
  return (
    <span title={tip} className={`inline-flex items-center gap-1.5 whitespace-nowrap text-[11px] font-medium text-fg-3 ${className}`}>
      <Search className="h-3 w-3 shrink-0" aria-hidden />
      <span>Tavily{kind ? <span className="text-muted"> · {kind}</span> : null}</span>
    </span>
  )
}

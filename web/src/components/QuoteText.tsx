import type { ReactNode } from 'react'
import { latexRows, matchesTarget } from '../latexRow'
import { Disclosure } from './ui'

const MAX_ROWS = 4

// A quote from a paper or repo. A LaTeX table row shows as its cells (the claimed value marked); anything else shows as the given plain
// fallback. The exact original text always stays one click away, because the quote is the evidence.
export function QuoteText({ quote, target, className = '', fallback }: { quote?: string; target?: number; className?: string; fallback: ReactNode }) {
  const rows = latexRows(quote)
  if (rows.length === 0) return <>{fallback}</>
  return (
    <div className={`border-l-2 border-line pl-3 ${className}`}>
      <div className="grid gap-1.5">
        {rows.slice(0, MAX_ROWS).map((cells, r) => (
          <div key={r} className="flex flex-wrap gap-1 font-mono text-[11px] leading-relaxed">
            {cells.map((c, i) => matchesTarget(c, target)
              ? <span key={i} className="whitespace-nowrap rounded bg-raised px-1.5 py-0.5 font-semibold text-fg-strong underline decoration-muted underline-offset-2">{c}</span>
              : <span key={i} className={`whitespace-nowrap rounded px-1.5 py-0.5 ${i === 0 ? 'bg-raised/60 text-fg-3' : 'text-muted'}`}>{c}</span>)}
          </div>
        ))}
      </div>
      <Disclosure summary="Show source text" className="mt-2" summaryClassName="text-[11px] text-muted hover:text-fg-3">
        <pre className="m-0 mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted" title="The exact quote from the source">{quote}</pre>
      </Disclosure>
    </div>
  )
}

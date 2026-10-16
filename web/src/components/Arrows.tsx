import type { ReactNode } from 'react'
import { ArrowRight } from 'lucide-react'
import { arrowParts } from './arrowParts'

// Model- or paper-written text can carry a unicode arrow. The UI never shows the glyph: it shows the lucide ArrowRight, sized to the text
// (1em), vertically centred, hidden from screen readers. Pure split in arrowParts (arrowParts.ts, tested in arrows.test.ts).

export function withArrows(text: string, keyPrefix: string | number = 0): ReactNode[] {
  const parts = arrowParts(text)
  const out: ReactNode[] = []
  parts.forEach((p, i) => {
    if (i > 0) out.push(<ArrowRight key={`a${keyPrefix}-${i}`} aria-hidden className="mx-[0.35em] inline-block h-[1em] w-[1em] align-[-0.125em] opacity-70" />)
    if (p) out.push(p)
  })
  return out
}

// the same arrow as a standalone icon between two flex children
export function InlineArrow({ className = 'h-4 w-4' }: { className?: string }) {
  return <ArrowRight aria-hidden className={`shrink-0 self-center text-subtle ${className}`} />
}

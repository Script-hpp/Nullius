import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

// A long block that starts closed: one quiet line with a title and a short fact, a click opens the rest. Nothing is removed.
export function Fold({ title, meta, open = false, onOpenChange, id, children }: { title: string; meta?: string; open?: boolean; onOpenChange?: (open: boolean) => void; id?: string; children: ReactNode }) {
  return (
    <details id={id} open={open} onToggle={(e) => onOpenChange?.(e.currentTarget.open)} className="group scroll-mt-16 border-t border-line last:border-b">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-4 text-left [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="text-[15px] font-medium text-fg-2">{title}</span>
          {meta && <span className="ml-3 text-[12px] text-muted">{meta}</span>}
        </span>
        <ChevronDown className="h-4 w-4 shrink-0 text-muted transition-transform group-open:rotate-180" aria-hidden />
      </summary>
      <div className="pb-8 pt-1">{children}</div>
    </details>
  )
}

// Heading of a tab: a title and one sentence, nothing else.
export function TabIntro({ title, text }: { title: string; text: string }) {
  return (
    <div className="mb-8">
      <h2 className="font-grotesk m-0 text-2xl font-semibold tracking-[-0.03em] text-fg">{title}</h2>
      <p className="mb-0 mt-2 max-w-2xl text-[14px] leading-relaxed text-fg-4">{text}</p>
    </div>
  )
}

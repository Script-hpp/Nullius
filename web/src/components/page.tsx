import type { ReactNode } from 'react'
import { Eye, Info } from 'lucide-react'
import { GoArrow, SHIFT_BTN } from './ui'
import { Glossed } from './Glossed'

// Building blocks shared by every page: the intro every page starts with, and the "Next" every page ends with.

// "What you see here" (one plain sentence) and "What to look at first" (where to start). Visible text, never a tooltip.
export function PageIntro({ kicker, title, what, look, compact, head, children }: { kicker?: string; title: string; what: string; look: string; compact?: string; head?: ReactNode; children?: ReactNode }) {
  return (
    <header className="pb-6 pt-6 md:pb-10 md:pt-14">
      {kicker && <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">{kicker}</div>}
      <h1 className="font-grotesk m-0 mt-3 text-4xl font-semibold leading-[1.02] text-fg sm:text-5xl">{title}</h1>
      {head}
      {compact && <div className="mt-4 md:hidden">
        <p className="m-0 text-[14px] leading-relaxed text-fg-3"><Glossed text={compact} /></p>
        <details className="mt-3 text-[13px] text-muted">
          <summary className="cursor-pointer">How to use this page</summary>
          <p className="mt-3 leading-relaxed"><Glossed text={what} /></p>
          <p className="mt-2 leading-relaxed"><Glossed text={look} /></p>
        </details>
      </div>}
      <div className={`${compact ? 'hidden md:grid' : 'grid'} mt-8 gap-6 border-t border-line pt-8 md:grid-cols-2 md:gap-0`}>
        <div className="md:pr-12">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted"><Info className="h-3.5 w-3.5" />What you see here</div>
          <p className="mb-0 mt-3 max-w-md text-[15px] leading-relaxed text-fg-3"><Glossed text={what} /></p>
        </div>
        <div className="border-line md:border-l md:pl-12">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted"><Eye className="h-3.5 w-3.5" />What to look at first</div>
          <p className="mb-0 mt-3 max-w-md text-[15px] leading-relaxed text-fg-3"><Glossed text={look} /></p>
        </div>
      </div>
      {children}
    </header>
  )
}

// The guided way: every page ends with one clear next step.
export function NextStep({ title, text, label, href }: { title: string; text: string; label: string; href: string }) {
  return (
    <section className="mt-20 flex flex-wrap items-center justify-between gap-6 border-t border-line pt-10" aria-label="Next step">
      <div className="max-w-xl">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Next</div>
        <p className="font-grotesk m-0 mt-3 text-2xl font-semibold tracking-[-0.03em] text-fg">{title}</p>
        <p className="mb-0 mt-2 text-[15px] leading-relaxed text-fg-4"><Glossed text={text} /></p>
      </div>
      <a href={href} className={`${SHIFT_BTN} inline-flex items-center gap-2 rounded-full bg-inverse px-6 py-3 text-sm font-medium text-on-inverse no-underline hover:bg-inverse-hover`}>
        {label} <GoArrow className="h-4 w-4 shrink-0" />
      </a>
    </section>
  )
}

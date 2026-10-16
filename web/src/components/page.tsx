import type { ReactNode } from 'react'
import { Chevron, GoArrow, SHIFT_BTN } from './ui'
import { Glossed } from './Glossed'

// Building blocks shared by every page: the intro every page starts with, and the "Next" every page ends with.

// Every page starts the same way: small label, one H1, one short line under it (max ~15 words). Anything longer sits in a closed "About this page".
export function PageIntro({ kicker, title, sub, about, head, children }: { kicker?: string; title: string; sub: string; about?: ReactNode; head?: ReactNode; children?: ReactNode }) {
  return (
    <header className="pb-6 pt-6 md:pb-8 md:pt-12">
      {kicker && <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">{kicker}</div>}
      <h1 className="font-grotesk m-0 mt-3 text-balance text-4xl font-semibold leading-[1.04] text-fg sm:text-5xl">{title}</h1>
      <p className="m-0 mt-4 max-w-2xl text-pretty text-[16px] leading-relaxed text-fg-4 sm:text-[17px]"><Glossed text={sub} /></p>
      {head}
      {about && (
        <details className="mt-4 text-[13px] text-muted">
          <summary className="nx-summary inline-flex cursor-pointer items-center gap-1.5 text-muted hover:text-fg-3">About this page<Chevron className="h-3.5 w-3.5" /></summary>
          <div className="mt-3 max-w-2xl space-y-2 leading-relaxed text-fg-4">{about}</div>
        </details>
      )}
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

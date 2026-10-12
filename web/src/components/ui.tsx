import type { ReactNode } from 'react'
import { ArrowRight, CheckCircle2, ChevronDown, MinusCircle, XCircle, Info, HelpCircle } from 'lucide-react'
import { TONE_TEXT, type Tone } from '../labels'

const toneConfig: Record<Tone, { color: string; icon: ReactNode }> = {
  green: { color: TONE_TEXT.green, icon: <CheckCircle2 className="w-4 h-4" /> },
  amber: { color: TONE_TEXT.amber, icon: <MinusCircle className="w-4 h-4" /> },
  red: { color: TONE_TEXT.red, icon: <XCircle className="w-4 h-4" /> },
  gray: { color: TONE_TEXT.gray, icon: <Info className="w-4 h-4" /> },
  violet: { color: TONE_TEXT.violet, icon: <HelpCircle className="w-4 h-4" /> }
}

export function toPercent(val: string | number | null | undefined): string {
  if (val === null || val === undefined || val === '') return '–'
  const n = Number(val)
  if (Number.isNaN(n)) return String(val)
  if (n <= 2.0 && n > 0) return (n * 100).toFixed(1) + '%'
  return n.toFixed(1) + '%'
}

export function Badge({ children, tone = 'gray', title, icon }: { children: ReactNode; tone?: Tone; title?: string; icon?: ReactNode }) {
  const cfg = toneConfig[tone] || toneConfig.gray
  return (
    <span title={title} className={`inline-flex items-center gap-1.5 text-xs font-medium tracking-wider uppercase ${cfg.color}`}>
      {icon ?? cfg.icon}
      {children}
    </span>
  )
}

export function Section({ title, hint, children, className = '', style }: { title: string; hint?: string; children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <section className={`p-6 bg-surface-2 border border-line/80 rounded-xl ${className}`} style={style}>
      <div className="mb-5 flex items-baseline justify-between gap-4 border-b border-line/50 pb-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-fg-4">{title}</h2>
        {hint && <p className="text-subtle text-[11px] uppercase tracking-wider">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-muted text-[10px] uppercase tracking-wider font-semibold mb-1">{label}</div>
      <div className="font-mono text-2xl font-semibold text-fg">{value}</div>
      {sub && <div className="text-subtle text-[11px] mt-1">{sub}</div>}
    </div>
  )
}

export const fmt = (x: number | null | undefined, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))

// ---------------------------------------------------------------------------------------------
// Interaction conventions. Every clickable thing is exactly one of these; use the primitive, not a one-off icon.
//
//   1. Internal link (another page or section of this site): text + open arrow on the RIGHT (ArrowRight, always).
//      Text link  -> <GoLink href>   (the arrow slides right on hover)
//      Button-like (custom classes) -> className={SHIFT_BTN}, arrow <GoArrow/> inside (the whole button slides right)
//   2. External link (another site, new tab): boxed diagonal arrow on the right -> <ExtLink href>
//      Only the arrow moves, outward along its diagonal. Use <ExtArrow/> if you build the element yourself,
//      with className "nx-ext" on the link.
//   3. Expand / collapse: <details>/<summary> with <Chevron/> (open chevron, rotates when open). No underline.
//      Ready-made: <Disclosure summary="...">.
//   4. Switch tab / reveal in place (stays on this page): quiet secondary button, NO arrow -> className={TAB_BTN}.
//   5. Non-interactive text: no pointer cursor, no underline, no hover change. A tooltip-only info icon uses
//      cursor-help (see components/Hint.tsx).
//
// Icons next to text: wrap both in an element with INLINE (flex + items-center + gap) so they stay vertically centered.
// Icon sizes: ICON_SM (12 px, small links) / ICON_MD (14 px, body links) / 16 px for buttons.
// Motion: 150-250 ms ease-out, disabled under prefers-reduced-motion (see index.css, nx-* classes).
// ---------------------------------------------------------------------------------------------
export const INLINE = 'inline-flex items-center gap-1.5'
export const ICON_SM = 'h-3 w-3 shrink-0'
export const ICON_MD = 'h-3.5 w-3.5 shrink-0'
export const SHIFT_BTN = 'nx-shift'
// In-page action (switches a tab or opens something on the same page): outlined button, no link arrow, no underline.
export const TAB_BTN = 'inline-flex cursor-pointer items-center rounded-lg border border-line-strong bg-transparent px-4 py-2 text-[13px] font-medium text-fg no-underline transition-colors hover:bg-raised'
// Clickable cards: border darkens, small lift and shadow (the lift is dropped under prefers-reduced-motion).
export const CARD_HOVER = 'transition-[border-color,box-shadow,transform] duration-200 ease-out hover:border-line-strong hover:shadow-md motion-safe:hover:-translate-y-0.5'
export const LINK_TEXT = 'text-fg underline underline-offset-4 hover:text-fg-max'

export function GoArrow({ className = ICON_MD }: { className?: string }) {
  return <ArrowRight className={`nx-arrow ${className}`} aria-hidden />
}

export function Chevron({ className = 'h-4 w-4' }: { className?: string }) {
  return <ChevronDown className={`nx-chev shrink-0 ${className}`} aria-hidden />
}

// Lucide's square-arrow-out-up-right, split in two: the box stays, the arrow (.nx-ext-arrow) moves on hover of an .nx-ext parent.
export function ExtArrow({ className = ICON_MD }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"
      className={className} style={{ overflow: 'visible' }} aria-hidden focusable="false">
      <path d="M21 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6" />
      <g className="nx-ext-arrow"><path d="m21 3-9 9" /><path d="M15 3h6v6" /></g>
    </svg>
  )
}

type LinkProps = { href: string; children: ReactNode; className?: string; title?: string; onClick?: (e: React.MouseEvent<HTMLAnchorElement>) => void }

export function GoLink({ href, children, className = LINK_TEXT, title, onClick }: LinkProps) {
  return <a href={href} title={title} onClick={onClick} className={`nx-go ${INLINE} ${className}`}>{children}<GoArrow /></a>
}

export function ExtLink({ href, children, className = LINK_TEXT, title }: LinkProps) {
  return <a href={href} title={title} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} ${className}`}>{children}<ExtArrow /></a>
}

// <details> with the standard summary: label on the left, chevron on the right of the label. Text is never underlined.
export function Disclosure({ summary, children, className = '', summaryClassName = 'text-fg-3 hover:text-fg' }: { summary: ReactNode; children: ReactNode; className?: string; summaryClassName?: string }) {
  return (
    <details className={className}>
      <summary className={`nx-summary ${INLINE} cursor-pointer text-[14px] no-underline ${summaryClassName}`}>{summary}<Chevron /></summary>
      {children}
    </details>
  )
}

// One fixed-width cell for a legend swatch, so the texts of all legend lines start at the same x.
export function LegendSwatch({ children }: { children: ReactNode }) {
  return <span className="mt-[3px] flex h-4 w-6 shrink-0 items-center justify-center" aria-hidden>{children}</span>
}

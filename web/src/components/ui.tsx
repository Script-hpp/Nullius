import type { ReactNode } from 'react'
import { TONE_VAR, type Tone } from '../labels'

export function Badge({ children, tone = 'gray', title }: { children: ReactNode; tone?: Tone; title?: string }) {
  const c = TONE_VAR[tone]
  return (
    <span title={title} className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium"
      style={{ color: c, background: `color-mix(in srgb, ${c} 14%, transparent)`, border: `1px solid color-mix(in srgb, ${c} 35%, transparent)` }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />
      {children}
    </span>
  )
}

export function Section({ title, hint, children, className = '' }: { title: string; hint?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`panel p-5 ${className}`}>
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider" style={{ color: 'var(--blue)' }}>{title}</h2>
        {hint && <p className="muted text-xs">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="muted text-xs uppercase tracking-wide">{label}</div>
      <div className="mono text-2xl font-semibold">{value}</div>
      {sub && <div className="muted text-xs">{sub}</div>}
    </div>
  )
}

export const fmt = (x: number | null | undefined, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))

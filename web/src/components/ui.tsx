import type { ReactNode } from 'react'
import { CheckCircle2, MinusCircle, XCircle, Info, HelpCircle } from 'lucide-react'
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

export function Badge({ children, tone = 'gray', title }: { children: ReactNode; tone?: Tone; title?: string }) {
  const cfg = toneConfig[tone] || toneConfig.gray
  return (
    <span title={title} className={`inline-flex items-center gap-1.5 text-xs font-medium tracking-wider uppercase ${cfg.color}`}>
      {cfg.icon}
      {children}
    </span>
  )
}

export function Section({ title, hint, children, className = '', style }: { title: string; hint?: string; children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <section className={`p-6 bg-[#0c0c0c] border border-zinc-800/80 rounded-xl ${className}`} style={style}>
      <div className="mb-5 flex items-baseline justify-between gap-4 border-b border-zinc-800/50 pb-3">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">{title}</h2>
        {hint && <p className="text-zinc-600 text-[11px] uppercase tracking-wider">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-zinc-500 text-[10px] uppercase tracking-wider font-semibold mb-1">{label}</div>
      <div className="font-mono text-2xl font-semibold text-zinc-100">{value}</div>
      {sub && <div className="text-zinc-600 text-[11px] mt-1">{sub}</div>}
    </div>
  )
}

export const fmt = (x: number | null | undefined, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))

import type { ReactNode } from 'react'
import { CheckCircle2, AlertCircle, XCircle, Info, HelpCircle } from 'lucide-react'
import type { Tone } from '../labels'

const toneConfig: Record<Tone, { color: string; bg: string; border: string; icon: ReactNode }> = {
  green: { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
  amber: { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', icon: <AlertCircle className="w-3.5 h-3.5" /> },
  red: { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', icon: <XCircle className="w-3.5 h-3.5" /> },
  gray: { color: 'text-zinc-400', bg: 'bg-zinc-500/10', border: 'border-zinc-500/20', icon: <Info className="w-3.5 h-3.5" /> },
  violet: { color: 'text-violet-400', bg: 'bg-violet-500/10', border: 'border-violet-500/20', icon: <HelpCircle className="w-3.5 h-3.5" /> }
}

export function Badge({ children, tone = 'gray', title }: { children: ReactNode; tone?: Tone; title?: string }) {
  const cfg = toneConfig[tone] || toneConfig.gray
  return (
    <span title={title} className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border ${cfg.bg} ${cfg.color} ${cfg.border}`}>
      {cfg.icon}
      {children}
    </span>
  )
}

export function Section({ title, hint, children, className = '', style }: { title: string; hint?: string; children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <section className={`p-6 bg-zinc-900 border border-zinc-800 rounded-xl ${className}`} style={style}>
      <div className="mb-5 flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-300">{title}</h2>
        {hint && <p className="text-zinc-500 text-xs">{hint}</p>}
      </div>
      {children}
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-zinc-500 text-xs uppercase tracking-wide font-medium mb-1">{label}</div>
      <div className="font-mono text-2xl font-semibold text-zinc-100">{value}</div>
      {sub && <div className="text-zinc-500 text-xs mt-0.5">{sub}</div>}
    </div>
  )
}

export const fmt = (x: number | null | undefined, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))

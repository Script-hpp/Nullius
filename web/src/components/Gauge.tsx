import { Bar, BarChart, CartesianGrid, Cell, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ATTEMPT, TONE_VAR, de2en } from '../labels'
import type { Json } from '../data'
import { toPercent } from './ui'

export function AttemptChart({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const attempts = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId && a.measured !== null && a.measured !== undefined)
  if (!claim || attempts.length === 0) return <p className="text-sm text-zinc-500 italic">No valid measurements.</p>
  const data = attempts.map((a: Json, i: number) => ({ name: `#${a.seed}`, i, value: a.measured, status: a.status }))
  const vals = data.map((d: Json) => d.value as number)
  const mean = vals.reduce((s: number, v: number) => s + v, 0) / vals.length
  const baselines: [string, number][] = []
  for (const [k, v] of Object.entries(dossier.baselines ?? {})) {
    const val = (v as Json)[claim.metric]
    if (typeof val === 'number') baselines.push([k, val])
  }
  const lo = claim.target - claim.tolerance, hi = claim.target + claim.tolerance
  const pad = (Math.max(...vals, hi) - Math.min(...vals, lo)) * 0.35 || 0.02
  const domain: [number, number] = [Math.min(...vals, lo) - pad, Math.max(...vals, hi) + pad]
  const shownBase = baselines.filter(([, v]) => v >= domain[0] && v <= domain[1])
  
  const f = (v: number) => toPercent(v)

  return (
    <div className="flex flex-col h-full">
      <div style={{ width: '100%', height: 260 }} className="flex-1">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#71717a', fontSize: 11, fontFamily: 'monospace' }} tickLine={false} axisLine={{ stroke: '#27272a' }} interval="preserveStartEnd" />
            <YAxis domain={domain} tickFormatter={f} tick={{ fill: '#71717a', fontSize: 11, fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={65} allowDataOverflow />
            <ReferenceArea y1={lo} y2={hi} fill="#34d399" fillOpacity={0.08} stroke="#34d399" strokeOpacity={0.0} />
            <ReferenceLine y={claim.target} stroke="#34d399" strokeWidth={1} strokeDasharray="4 4" label={{ value: `Target: ${f(claim.target)}`, fill: '#34d399', fontSize: 11, position: 'insideTopRight', fontFamily: 'monospace' }} />
            <ReferenceLine y={mean} stroke="#60a5fa" strokeDasharray="2 2" label={{ value: `Mean: ${f(mean)}`, fill: '#60a5fa', fontSize: 11, position: 'insideBottomRight', fontFamily: 'monospace' }} />
            {shownBase.map(([k, v]) => <ReferenceLine key={k} y={v} stroke="#71717a" strokeDasharray="2 4" label={{ value: k, fill: '#71717a', fontSize: 10, position: 'insideBottomLeft' }} />)}
            <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ background: '#0a0a0a', border: '1px solid #27272a', borderRadius: 8, color: '#f4f4f5' }}
              formatter={(v) => [f(Number(v)), 'Measured']} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={700}>
              {data.map((d: Json, i: number) => <Cell key={i} fill={TONE_VAR[de2en(ATTEMPT, d.status).tone]} fillOpacity={0.9} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-zinc-500 mt-4 text-[11px] leading-relaxed border-t border-zinc-800/50 pt-3">
        Green band = target ± tolerance ({f(lo)} to {f(hi)}). Bars = individual sandbox executions (seeds). The final verdict is based on the <b>mean</b> of all stable runs.
        {shownBase.length < baselines.length && ' Trivial baselines outside the range are omitted: ' + baselines.filter((b) => !shownBase.includes(b)).map(([k, v]) => `${k} ${f(v)}`).join(', ') + '.'}
      </p>
    </div>
  )
}

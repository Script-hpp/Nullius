import { Bar, BarChart, CartesianGrid, Cell, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Json } from '../data'
import { toPercent } from './ui'

export function AttemptChart({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const attempts = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId && a.measured !== null && a.measured !== undefined)
  if (!claim || attempts.length === 0) return <p className="text-sm text-muted italic">No valid measurements.</p>
  const data = [...attempts].sort((a: Json, b: Json) => a.seed - b.seed)   // seed order #0..#n (stable: retries of a seed stay in run order)
    .map((a: Json, i: number) => ({ name: `#${a.seed}`, i, value: a.measured, status: a.status }))
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
  
  const f = (v: number) => (claim.metric === 'accuracy' ? toPercent(v) : v.toFixed(3))

  return (
    <div className="flex flex-col flex-1">
      <div style={{ width: '100%', height: 260, minHeight: 260 }} className="flex-1 mt-2">
        <ResponsiveContainer width="100%" height="100%" minHeight={260}>
          <BarChart data={data} margin={{ top: 20, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: 'var(--chart-axis)', fontSize: 11, fontFamily: 'monospace' }} tickLine={false} axisLine={{ stroke: 'var(--chart-grid)' }} interval="preserveStartEnd" />
            <YAxis domain={domain} tickFormatter={f} tick={{ fill: 'var(--chart-axis)', fontSize: 11, fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={65} allowDataOverflow />
            
            <ReferenceArea y1={lo} y2={hi} fill="var(--chart-target)" fillOpacity={0.08} stroke="var(--chart-target)" strokeOpacity={0.0} />
            <ReferenceLine y={claim.target} stroke="var(--chart-target)" strokeWidth={1} strokeDasharray="5 5" label={{ value: `Target: ${f(claim.target)}`, fill: 'var(--chart-target)', fontSize: 11, position: 'insideBottomLeft', fontFamily: 'monospace', offset: 8 }} />
            
            <ReferenceLine y={mean} stroke="var(--chart-mean)" strokeDasharray="3 3" label={{ value: `Mean: ${f(mean)}`, fill: 'var(--chart-mean)', fontSize: 11, position: 'insideTopRight', fontFamily: 'monospace', offset: 8 }} />
            
            {shownBase.map(([k, v]) => <ReferenceLine key={k} y={v} stroke="var(--chart-gray)" strokeDasharray="2 4" label={{ value: k, fill: 'var(--chart-axis)', fontSize: 10, position: 'insideBottomRight' }} />)}
            
            <Tooltip cursor={{ fill: 'var(--chart-cursor)' }} contentStyle={{ background: 'var(--chart-tooltip-bg)', border: '1px solid var(--chart-tooltip-line)', borderRadius: 8, color: 'var(--chart-tooltip-fg)' }}
              formatter={(v) => [f(Number(v)), 'Measured']} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={700}>
              {data.map((_: Json, i: number) => <Cell key={i} fill="var(--c-fill-mid)" fillOpacity={0.55} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted leading-relaxed mt-4 pt-4 border-t border-line/50">
        One bar per seed (one run in its own sandbox branch). Band = claimed value ± tolerance ({f(lo)} to {f(hi)}). The verdict uses the <b>mean</b> of all seeds, so single bars may fall outside the band.
        {shownBase.length < baselines.length && ' Trivial baselines outside the range are omitted: ' + baselines.filter((b) => !shownBase.includes(b)).map(([k, v]) => `${k} ${f(v)}`).join(', ') + '.'}
      </p>
    </div>
  )
}

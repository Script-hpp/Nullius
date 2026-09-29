import { Bar, BarChart, CartesianGrid, Cell, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ATTEMPT, TONE_VAR, de2en } from '../labels'
import type { Json } from '../data'

// Messwerte je Seed mit Toleranzband um den Zielwert, Mittelwert und Baselines.
export function AttemptChart({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const attempts = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId && a.measured !== null && a.measured !== undefined)
  if (!claim || attempts.length === 0) return <p className="muted text-sm">No valid measurements.</p>
  const data = attempts.map((a: Json, i: number) => ({ name: `#${a.seed}`, i, value: a.measured, status: a.status }))
  const vals = data.map((d: Json) => d.value as number)
  const mean = vals.reduce((s: number, v: number) => s + v, 0) / vals.length
  const isAcc = claim.metric === 'accuracy'
  const baselines: [string, number][] = []
  for (const [k, v] of Object.entries(dossier.baselines ?? {})) {
    const val = (v as Json)[claim.metric]
    if (typeof val === 'number') baselines.push([k, val])
  }
  const lo = claim.target - claim.tolerance, hi = claim.target + claim.tolerance
  const pad = (Math.max(...vals, hi) - Math.min(...vals, lo)) * 0.35 || 0.02
  const domain: [number, number] = [Math.min(...vals, lo) - pad, Math.max(...vals, hi) + pad]
  // Baselines nur zeigen, wenn sie in den Bereich fallen (sonst staucht eine Linie bei 0,13 das Diagramm)
  const shownBase = baselines.filter(([, v]) => v >= domain[0] && v <= domain[1])
  const f = (v: number) => (isAcc ? `${(v * 100).toFixed(1)}%` : v.toFixed(3))
  return (
    <div>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 11 }} tickLine={false} axisLine={{ stroke: 'var(--line)' }} interval="preserveStartEnd" />
            <YAxis domain={domain} tickFormatter={f} tick={{ fill: 'var(--muted)', fontSize: 11 }} tickLine={false} axisLine={false} width={52} allowDataOverflow />
            <ReferenceArea y1={lo} y2={hi} fill="var(--green)" fillOpacity={0.12} stroke="var(--green)" strokeOpacity={0.4} strokeDasharray="3 3" />
            <ReferenceLine y={claim.target} stroke="var(--green)" strokeWidth={1.5} label={{ value: `target ${f(claim.target)}`, fill: 'var(--green)', fontSize: 11, position: 'insideTopRight' }} />
            <ReferenceLine y={mean} stroke="var(--blue)" strokeDasharray="5 4" label={{ value: `mean ${f(mean)}`, fill: 'var(--blue)', fontSize: 11, position: 'insideBottomRight' }} />
            {shownBase.map(([k, v]) => <ReferenceLine key={k} y={v} stroke="var(--muted)" strokeDasharray="2 4" label={{ value: k, fill: 'var(--muted)', fontSize: 10, position: 'insideBottomLeft' }} />)}
            <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={{ background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: 8, color: 'var(--text)' }}
              formatter={(v) => [f(Number(v)), 'measured']} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={700}>
              {data.map((d: Json, i: number) => <Cell key={i} fill={TONE_VAR[de2en(ATTEMPT, d.status).tone]} fillOpacity={0.85} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="muted mt-2 text-xs">
        Green band = target ± tolerance ({f(lo)} to {f(hi)}). Bars = one run each; colour = status. The verdict rests on the <b>mean</b> over all runs, not on single runs.
        {shownBase.length < baselines.length && ' Trivial baselines outside the range are omitted: ' + baselines.filter((b) => !shownBase.includes(b)).map(([k, v]) => `${k} ${f(v)}`).join(', ') + '.'}
      </p>
    </div>
  )
}

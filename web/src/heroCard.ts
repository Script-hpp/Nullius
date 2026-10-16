import { easeOut } from './countUp.ts'
import { formatMetric } from './metricFormat.ts'

// Pure data for the result card in the start page hero. Everything comes from the exported example run and its catalog entry; nothing is typed in.

export type HeroCard = {
  metric: string | undefined
  name: string          // short method name, "FITS"
  line: string          // what is measured, "FITS · forecast error on ETTh1"
  hint: string          // "lower is better"
  spread: string | null // "± 0.00005", the spread of the measured value over seeds
  claimed: string
  measured: string
  measuredValue: number
  urteil: string
  meta: string          // "3 seeds · recorded 14 Oct 2026 · live Tavily calls · $0.31"
}

type Claim = { metric?: string; target?: number }
type Verdict = { urteil?: string; measured_mean?: number; measured_std?: number | null }
type Source = { kind?: string; mode?: string }
type Entry = { claim?: { method?: string }; dataset?: string; cost_gross_usd?: number | null } | undefined

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// "20261014-080542" -> "14 Oct 2026"; anything else -> null
export function recordedDate(ts: string | undefined): string | null {
  const m = /^(\d{4})(\d{2})(\d{2})-/.exec(ts ?? '')
  if (!m) return null
  const month = MONTHS[Number(m[2]) - 1]
  return month ? `${Number(m[3])} ${month} ${m[1]}` : null
}

const errorWord = (metric: string | undefined) => (metric === 'accuracy' ? 'accuracy' : metric === 'mae' ? 'forecast error (MAE)' : 'forecast error')

export function heroCard(
  run: { ref: { title?: string; timestamp?: string }; events?: { type?: string }[]; dossier?: { claims?: Claim[]; verdicts?: Verdict[]; attempts?: { seed?: number; measured?: number | null }[]; sources?: Source[] } } | null,
  entry: Entry,
  seeds: number,
): HeroCard | null {
  const claim = run?.dossier?.claims?.[0]
  const v = run?.dossier?.verdicts?.[0]
  if (!claim || !v || claim.target == null || v.measured_mean == null || !v.urteil) return null
  const name = entry?.claim?.method ?? run!.ref.title?.split(':')[0]?.trim() ?? 'Paper'
  const where = entry?.dataset ? ` on ${entry.dataset}` : ''
  const target = formatMetric(claim.metric, claim.target)
  const parts: string[] = [`${seeds} seeds`]
  const date = recordedDate(run!.ref.timestamp)
  if (date) parts.push(`recorded ${date}`)
  const live = run!.dossier?.sources?.some((s) => s.mode === 'live')
  const calls = run!.events?.filter((e) => e.type === 'tavily_call').length   // undefined: events not loaded, so no count
  if (calls) parts.push(`${calls} ${live ? 'live ' : ''}Tavily ${calls === 1 ? 'call' : 'calls'}`)
  else if (live && calls === undefined) parts.push('live Tavily calls')
  if (entry?.cost_gross_usd != null) parts.push(`$${entry.cost_gross_usd.toFixed(2)}`)
  const share = claim.metric === 'accuracy'
  return {
    metric: claim.metric,
    name,
    line: `${name} · ${errorWord(claim.metric)}${where}`,
    hint: share ? 'higher is better' : 'lower is better',
    spread: v.measured_std == null ? null : `± ${spreadText(v.measured_std, share)}`,
    claimed: target,
    measured: formatMetric(claim.metric, v.measured_mean),
    measuredValue: v.measured_mean,
    urteil: v.urteil,
    meta: parts.join(' · '),
  }
}

// Two significant digits for a small spread (0.000046 -> 0.00005), one decimal for percent points.
function spreadText(std: number, share: boolean): string {
  if (share) return (std * 100).toFixed(1)
  return std > 0 && std < 0.001 ? std.toPrecision(1) : std.toFixed(4)
}

// Value shown `elapsed` ms into a `duration` ms ease-out ramp from 0 to `target` (decimals kept, unlike countAt).
export function rampValue(target: number, elapsed: number, duration: number): number {
  if (!(duration > 0) || elapsed >= duration) return target
  return target * easeOut(elapsed / duration)
}

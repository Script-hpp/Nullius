import { easeOut } from './countUp.ts'
import { formatMetric } from './metricFormat.ts'

// Pure data for the result card in the start page hero. Everything comes from the exported example run and its catalog entry; nothing is typed in.

export type HeroCard = {
  metric: string | undefined
  name: string          // short method name, "FITS"
  line: string          // plain-words claim, "FITS: forecast error 0.372 on ETTh1"
  claimed: string
  measured: string
  measuredValue: number
  urteil: string
  meta: string          // "3 seeds · recorded 14 Oct 2026 · live Tavily calls · $0.31"
}

type Claim = { metric?: string; target?: number }
type Verdict = { urteil?: string; measured_mean?: number }
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
  run: { ref: { title?: string; timestamp?: string }; dossier?: { claims?: Claim[]; verdicts?: Verdict[]; attempts?: { seed?: number; measured?: number | null }[]; sources?: Source[] } } | null,
  entry: Entry,
  seeds: number,
): HeroCard | null {
  const claim = run?.dossier?.claims?.[0]
  const v = run?.dossier?.verdicts?.[0]
  if (!claim || !v || claim.target == null || v.measured_mean == null || !v.urteil) return null
  const name = entry?.claim?.method ?? run!.ref.title?.split(':')[0]?.trim() ?? 'Paper'
  const where = entry?.dataset ? ` on ${entry.dataset}` : ''
  const target = formatMetric(claim.metric, claim.target)
  const said = claim.metric === 'accuracy' ? target : String(claim.target)   // the paper's own digits: 0.372, not 0.3720
  const parts: string[] = [`${seeds} seeds`]
  const date = recordedDate(run!.ref.timestamp)
  if (date) parts.push(`recorded ${date}`)
  if (run!.dossier?.sources?.some((s) => s.mode === 'live')) parts.push('live Tavily calls')
  if (entry?.cost_gross_usd != null) parts.push(`$${entry.cost_gross_usd.toFixed(2)}`)
  return {
    metric: claim.metric,
    name,
    line: claim.metric === 'accuracy' ? `${name}: ${said} ${errorWord(claim.metric)}${where}` : `${name}: ${errorWord(claim.metric)} ${said}${where}`,
    claimed: target,
    measured: formatMetric(claim.metric, v.measured_mean),
    measuredValue: v.measured_mean,
    urteil: v.urteil,
    meta: parts.join(' · '),
  }
}

// Value shown `elapsed` ms into a `duration` ms ease-out ramp from 0 to `target` (decimals kept, unlike countAt).
export function rampValue(target: number, elapsed: number, duration: number): number {
  if (!(duration > 0) || elapsed >= duration) return target
  return target * easeOut(elapsed / duration)
}

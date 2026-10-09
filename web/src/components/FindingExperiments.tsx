import { ArrowRight, ChevronDown, ExternalLink, FlaskConical, GitBranch, Ruler, TextQuote } from 'lucide-react'
import { Badge } from './ui'
import type { Json } from '../data'
import { STATUS, TYPE_LABEL, discardedFindings, experimentCards, funnel, provenance, type ExperimentCard, type Measure } from '../objections'

const host = (url: string) => { try { const u = new URL(url); return u.host + u.pathname.replace(/\/$/, '') } catch { return url } }
const f = (x: number, d = 4) => x.toFixed(d)
const COL = 'min-w-0'
const COL_HEAD = 'mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-widest text-muted'

// One shared axis: the paper target with its tolerance band, each measured value a dot with its spread, so words and numbers compare by eye.
function MeasureRows({ m }: { m: Measure }) {
  const vals = [...m.points.flatMap((p) => [p.value - (p.spread ?? 0), p.value + (p.spread ?? 0)]), ...(m.target !== null ? [m.target - (m.tolerance ?? 0), m.target + (m.tolerance ?? 0)] : [])]
  if (!vals.length) return null
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo || 0.01) * 0.15
  const a = lo - pad, b = hi + pad
  const x = (v: number) => `${(((v - a) / (b - a)) * 100).toFixed(2)}%`
  const w = (v: number) => `${((v / (b - a)) * 100).toFixed(2)}%`
  return (
    <div className="mt-3" data-testid="measure-strip">
      <div className="relative">
        {m.target !== null && m.tolerance !== null && <div className="absolute inset-y-0 rounded-sm bg-tone-green/10" style={{ left: x(m.target - m.tolerance), width: w(2 * m.tolerance) }} />}
        {m.target !== null && <div className="absolute inset-y-0 border-l border-dashed border-fg-4" style={{ left: x(m.target) }} />}
        {m.points.map((p) => {
          const color = p.kind === 'reported' ? 'bg-tone-amber-solid' : p.kind === 'ours' ? 'bg-tone-green-solid' : 'bg-fg-4'
          return (
            <div key={p.label} className="relative h-7">
              <div className="absolute inset-x-0 top-1/2 h-px bg-line/40" />
              {p.spread ? <div className={`absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full opacity-60 ${color}`} style={{ left: x(p.value - p.spread), width: w(2 * p.spread) }} /> : null}
              <div className={`absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ${color}`} style={{ left: x(p.value) }} />
            </div>
          )
        })}
      </div>
      <dl className="mt-2 space-y-1 text-[12px]">
        {m.points.map((p) => (
          <div key={p.label} className="flex items-baseline justify-between gap-3">
            <dt className="text-fg-4">{p.label}</dt>
            <dd className="font-mono text-fg-2">{f(p.value)}{p.spread ? <span className="text-muted"> ± {f(p.spread)}</span> : null}</dd>
          </div>
        ))}
        {m.target !== null && (
          <div className="flex items-baseline justify-between gap-3 border-t border-line/40 pt-1">
            <dt className="text-fg-4">Paper target</dt>
            <dd className="font-mono text-fg-2">{f(m.target, 3)}{m.tolerance !== null ? <span className="text-muted"> ± {f(m.tolerance, 3)}</span> : null}</dd>
          </div>
        )}
      </dl>
    </div>
  )
}

function Card({ c, src }: { c: ExperimentCard; src: string | null }) {
  const st = STATUS[c.status] ?? STATUS.offen
  return (
    <li className="border-t border-line py-6 first:border-t-0 first:pt-0" data-testid="experiment-card">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <Badge tone="violet" icon={<FlaskConical className="h-4 w-4" />}>{TYPE_LABEL[c.type] ?? c.type}</Badge>
        <Badge tone={st.tone}>{st.label}</Badge>
      </div>
      <div className="grid items-stretch gap-6 lg:grid-cols-[minmax(0,1.15fr)_auto_minmax(0,1fr)_auto_minmax(0,1.05fr)]">
        <div className={COL}>
          <div className={COL_HEAD}><TextQuote className="h-3.5 w-3.5" />Words · found by Tavily</div>
          <blockquote className="m-0 border-l-2 border-line-strong/60 py-1 pl-3 font-mono text-[12px] italic leading-relaxed text-fg-3">“{c.quote}”</blockquote>
          <a className="mt-3 inline-flex max-w-full items-center gap-1.5 text-[11px] text-tone-blue transition-colors hover:text-tone-blue-hover" href={c.url} target="_blank" rel="noreferrer">
            <ExternalLink className="h-3 w-3 shrink-0" /><span className="truncate">{host(c.url)}</span>
          </a>
          <p className="mt-1 text-[11px] text-subtle">Role 3, community search{src ? ` · ${src}` : ''}</p>
        </div>
        <ArrowRight className="hidden h-4 w-4 self-center text-muted lg:block" aria-hidden />
        <div className={COL}>
          <div className={COL_HEAD}><GitBranch className="h-3.5 w-3.5" />Experiment · own branch</div>
          <p className="text-[13px] font-medium leading-relaxed text-fg-2">{c.title}</p>
          <ul className="mt-2 space-y-1.5 text-[12px] leading-relaxed text-fg-4">{c.details.map((d, i) => <li key={i}>{d}</li>)}</ul>
        </div>
        <ArrowRight className="hidden h-4 w-4 self-center text-muted lg:block" aria-hidden />
        <div className={COL}>
          <div className={COL_HEAD}><Ruler className="h-3.5 w-3.5" />Measurement · harness</div>
          <p className="text-[13px] font-medium leading-relaxed text-fg-2">{c.outcome}</p>
          <MeasureRows m={c.measure} />
        </div>
      </div>
    </li>
  )
}

export const experimentCount = (dossier: Json, claimId: string) => experimentCards(dossier, claimId).length

// Compact preview for the overview: the funnel in one line, then one line per experiment. The full cards sit in "Sources & Tavily".
export function FindingPreview({ dossier, claimId, onOpen }: { dossier: Json; claimId: string; onOpen: () => void }) {
  const cards = experimentCards(dossier, claimId)
  if (!cards.length) return null
  const n = funnel(dossier, claimId)
  return (
    <div data-testid="finding-preview">
      <p className="m-0 font-mono text-[11px] text-muted">{n.found} statements found · {n.kept} kept as notes · {n.tested} tested · {n.dropped} dropped</p>
      <ul className="m-0 mt-3 list-none border-t border-line p-0">
        {cards.map((c) => {
          const st = STATUS[c.status] ?? STATUS.offen
          return (
            <li key={c.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-1 border-b border-line py-2.5">
              <span className="min-w-0 text-[13px] leading-relaxed text-fg-3"><span className="font-medium text-fg-2">{TYPE_LABEL[c.type] ?? c.type}</span><span className="text-fg-4"> · {c.outcome}</span></span>
              <Badge tone={st.tone}>{st.label}</Badge>
            </li>
          )
        })}
      </ul>
      <button type="button" onClick={onOpen} className="mt-3 inline-flex cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-[13px] font-medium text-fg-3 underline decoration-line-strong underline-offset-4 hover:text-fg">
        See the quotes, branches and measurements<ArrowRight className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

// Findings that became experiments (words next to numbers) and, folded away, the findings that did not and why. Reported only, never part of the verdict.
export function FindingExperiments({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const cards = experimentCards(dossier, claimId)
  if (!cards.length) return null   // older runs without objection checks: no card
  const dropped = discardedFindings(dossier, claimId)
  const n = funnel(dossier, claimId)
  const src = provenance(dossier.community)
  return (
    <div className="min-w-0">
      <p className="mb-4 mt-0 max-w-2xl text-[13px] leading-relaxed text-fg-4">
        Tavily found these statements about the paper. Each one that names something we can repeat or change ran in its own sandbox branch, and the independent harness measured it. Confirmed or refuted follows a rule fixed before the run, for this task only. Not part of the verdict.
      </p>
      <p className="mb-5 mt-0 font-mono text-[11px] text-muted" data-testid="funnel">
        {n.found} statements found · {n.kept} kept as notes · {n.tested} tested · {n.dropped} dropped
      </p>
      <ul className="m-0 list-none p-0" data-testid="objection-checks">{cards.map((c) => <Card key={c.id} c={c} src={src} />)}</ul>
      {dropped.length > 0 && (
        <details className="group mt-2 border-t border-line" data-testid="discarded">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 py-3 text-[13px] font-medium text-fg-3 [&::-webkit-details-marker]:hidden">
            <span>{dropped.length} finding{dropped.length === 1 ? '' : 's'} not turned into an experiment</span>
            <ChevronDown className="h-4 w-4 text-muted transition-transform group-open:rotate-180" />
          </summary>
          <ul className="m-0 list-none space-y-4 p-0 pb-2">
            {dropped.map((d) => (
              <li key={d.key} className="text-[12px] leading-relaxed">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <Badge tone="gray" icon={<span />}>{d.stage === 'dropped' ? 'Dropped' : 'Kept as text'}</Badge>
                  <span className="text-fg-4">{d.reason}</span>
                </div>
                <p className="m-0 truncate font-mono italic text-muted" title={d.quote}>“{d.quote}”</p>
                <a className="mt-0.5 inline-flex max-w-full items-center gap-1.5 text-[11px] text-tone-blue hover:text-tone-blue-hover" href={d.url} target="_blank" rel="noreferrer">
                  <ExternalLink className="h-3 w-3 shrink-0" /><span className="truncate">{host(d.url)}</span>
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

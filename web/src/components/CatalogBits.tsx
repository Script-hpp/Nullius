import { AlertCircle, ArrowRight, Info } from 'lucide-react'
import { Badge, toPercent } from './ui'
import { NOT_ASSESSABLE, OVERALL, PRELIMINARY, TONE_TEXT, catalogLabel, isPreliminary, translateWarning } from '../labels'
import { evidenceHref, fmtClaimed, fmtDelta, fmtMeasured, type CatalogEntry } from '../catalog'
import type { Json, RunData } from '../data'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-zinc-500'

export function PreliminaryChip() {
  return (
    <span title="The code was not checked against the method described in the paper, so this result says nothing about the paper itself."
      className="inline-flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-zinc-400">
      Preliminary
    </span>
  )
}

// Result of a checked catalog entry: axis-1 verdict, overall statement, preliminary marker.
export function CatalogVerdict({ entry }: { entry: CatalogEntry }) {
  const v = entry.verdict
  if (!v) return null
  const r = catalogLabel(v.urteil, v.gesamtaussage)
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <Badge tone={r.tone}>{r.label}</Badge>
      <span className="text-[11px] text-zinc-500">{OVERALL[v.gesamtaussage] ?? v.gesamtaussage}</span>
      {isPreliminary(v.gesamtaussage) && <PreliminaryChip />}
    </div>
  )
}

// "CLAIMED 82.92%  MEASURED 78.3% ± 1.8  −4.6 pp"
export function ClaimedMeasured({ entry }: { entry: CatalogEntry }) {
  const v = entry.verdict
  const delta = fmtDelta(entry)
  const r = v ? catalogLabel(v.urteil, v.gesamtaussage) : null
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <div className={label}>Claimed</div>
        <div className="mt-1 font-mono text-lg font-semibold text-zinc-300">{fmtClaimed(entry)}</div>
      </div>
      <div>
        <div className={label}>Measured</div>
        <div className={`mt-1 font-mono text-lg font-semibold ${r ? TONE_TEXT[r.tone] : 'text-zinc-300'}`}>{fmtMeasured(entry)}</div>
        {delta && <div className="mt-0.5 font-mono text-[11px] text-zinc-500">{delta.text} vs. claimed</div>}
      </div>
    </div>
  )
}

export function EvidenceLink({ entry, tab, children }: { entry: CatalogEntry; tab?: 'log'; children: React.ReactNode }) {
  const href = evidenceHref(entry, tab)
  if (!href) return null
  return (
    <a href={href} className="inline-flex items-center gap-1 text-zinc-300 hover:text-white">
      {children}<ArrowRight className="w-3.5 h-3.5" />
    </a>
  )
}

// Callout above the verdict of a catalog run: what limits the statement.
export function CatalogCallout({ run, claimId, gesamt }: { run: RunData; claimId: string; gesamt?: string }) {
  const d = run.dossier
  const warnings: string[] = (d.warnings ?? []).map(translateWarning)
  const ab: Json = d.ablations?.[claimId]
  const repairs = run.events.filter((e) => e.type === 'graph_repair' && e.claim_id === claimId)
  const pct = (x: number | null | undefined) => (x == null ? '–' : toPercent(x))
  const items: string[] = []
  if (ab?.verdict === 'Graph nicht genutzt') {
    items.push(`The code ignores the graph: accuracy with edges ${pct(ab.accuracy_full)}, without edges ${pct(ab.accuracy_ablated)}. It does not implement the method.`)
  }
  if (repairs.length) {
    for (const r of repairs) {
      const outcome = r.result === 'repariert' ? 'code rewritten' : r.result === 'crash' ? 'rewritten code did not run, kept the previous code' : 'no code returned, kept the previous code'
      items.push(`Graph repair, round ${r.runde}: code ${r.from_hash} ignored the graph (${pct(r.acc_full)} with edges, ${pct(r.acc_ablated)} without); ${outcome}.`)
    }
  } else if (d.graph_repairs?.[claimId]) {
    items.push(`Graph repairs: ${d.graph_repairs[claimId]}.`)
  }
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-5">
      <div className="flex items-start gap-3">
        <Info className="mt-0.5 w-4 h-4 shrink-0 text-zinc-400" />
        <div className="min-w-0">
          <p className="text-[13px] font-medium text-zinc-200">
            {isPreliminary(gesamt) ? `${PRELIMINARY}.` : gesamt === NOT_ASSESSABLE ? 'Not assessable: the method was not implemented by the code.' : gesamt ? `${OVERALL[gesamt] ?? gesamt}.` : 'Discovery candidate run once through the loop.'}
          </p>
          <p className="mt-1 text-[12px] leading-relaxed text-zinc-500">
            This is a measurement with our setup, not a finding about the paper. We do not claim the paper is wrong.
          </p>
          {(items.length > 0 || warnings.length > 0) && (
            <ul className="mt-3 space-y-1.5">
              {items.map((t, i) => (
                <li key={`i${i}`} className="flex items-start gap-2 text-[12px] leading-relaxed text-zinc-400">
                  <AlertCircle className="mt-0.5 w-3.5 h-3.5 shrink-0 text-zinc-500" />{t}
                </li>
              ))}
              {warnings.map((t, i) => (
                <li key={`w${i}`} className="flex items-start gap-2 text-[12px] leading-relaxed text-zinc-400">
                  <Info className="mt-0.5 w-3.5 h-3.5 shrink-0 text-zinc-500" />{t}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

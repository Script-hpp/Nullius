import { AlertCircle, Info } from 'lucide-react'
import { Badge, GoArrow, ICON_MD, INLINE, toPercent } from './ui'
import { NOT_ASSESSABLE, OVERALL, PRELIMINARY, TONE_TEXT, catalogLabel, isPreliminary, resultHint, translateWarning } from '../labels'
import { evidenceHref, fmtClaimed, fmtDelta, fmtMeasured, type CatalogEntry } from '../catalog'
import type { Json, RunData } from '../data'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'

export function PreliminaryChip() {
  return (
    <span title="The code was not checked against the method described in the paper, so this result says nothing about the paper itself."
      className="inline-flex items-center gap-1 rounded border border-line-strong px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-fg-4">
      Method not verified
    </span>
  )
}

// Result of a checked catalog entry: axis-1 verdict, overall statement, preliminary marker.
export function CatalogVerdict({ entry }: { entry: CatalogEntry }) {
  const v = entry.verdict
  if (!v) return null
  const r = catalogLabel(v.urteil, v.gesamtaussage)
  const redundant = v.gesamtaussage === v.urteil || v.gesamtaussage === NOT_ASSESSABLE   // the badge already says it
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <Badge tone={r.tone} title={resultHint(v.urteil)}>{r.label}</Badge>
      {isPreliminary(v.gesamtaussage) ? <PreliminaryChip /> : !redundant && <span className="text-[11px] text-muted">{OVERALL[v.gesamtaussage] ?? v.gesamtaussage}</span>}
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
        <div className="mt-1 font-mono text-lg font-semibold text-fg-3">{fmtClaimed(entry)}</div>
      </div>
      <div>
        <div className={label}>Measured</div>
        <div className={`mt-1 font-mono text-lg font-semibold ${r ? TONE_TEXT[r.tone] : 'text-fg-3'}`}>{fmtMeasured(entry)}</div>
        {delta && <div className="mt-0.5 font-mono text-[11px] text-muted">{delta.text} vs. claimed</div>}
      </div>
    </div>
  )
}

export function EvidenceLink({ entry, tab, children }: { entry: CatalogEntry; tab?: 'log'; children: React.ReactNode }) {
  const href = evidenceHref(entry, tab)
  if (!href) return null
  return (
    <a href={href} className={`nx-go ${INLINE} text-fg-3 hover:text-fg-max`}>
      {children}<GoArrow className={ICON_MD} />
    </a>
  )
}

// What the code or the run needs flagged: ablation result, graph repairs, run warnings. Plain strings, so the verdict card can count them.
export function catalogNotes(run: RunData, claimId: string): { items: string[]; warnings: string[] } {
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
  return { items, warnings }
}

// Notes about one run for the Limits tab: overall statement, then each flag on its own line. Flat, no frame.
export function CatalogCallout({ run, claimId, gesamt }: { run: RunData; claimId: string; gesamt?: string }) {
  const { items, warnings } = catalogNotes(run, claimId)
  return (
    <div>
      <p className="m-0 text-[14px] font-medium leading-relaxed text-fg-2">
        {isPreliminary(gesamt) ? `${PRELIMINARY}.` : gesamt === NOT_ASSESSABLE ? 'Not assessable: the method was not implemented by the code.' : gesamt ? `${OVERALL[gesamt] ?? gesamt}.` : 'Discovery candidate run once through the loop.'}
      </p>
      <p className="mb-0 mt-1 text-[13px] leading-relaxed text-muted">This is a measurement with our setup, not a finding about the paper. We do not claim the paper is wrong.</p>
      {(items.length > 0 || warnings.length > 0) && (
        <ul className="m-0 mt-4 list-none space-y-2 p-0">
          {items.map((t, i) => (
            <li key={`i${i}`} className="flex items-start gap-2 text-[13px] leading-relaxed text-fg-4"><AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />{t}</li>
          ))}
          {warnings.map((t, i) => (
            <li key={`w${i}`} className="flex items-start gap-2 text-[13px] leading-relaxed text-fg-4"><Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />{t}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

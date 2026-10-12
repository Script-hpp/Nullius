import { formatMetric } from '../metricFormat'
import { measuredAttempts } from '../measurements'
import { AttemptChart } from './Gauge'
import type { Json } from '../data'
import { targetSource } from '../targetSource'

const H3 = 'm-0 mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted'

// The sentence of the paper that holds the target number, quoted from the page Tavily read.
export function TargetEvidence({ claim, sources }: { claim: Json; sources?: Json[] }) {
  const source = targetSource(claim, sources)
  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">Target value</span>
        <span className="font-mono text-lg font-semibold text-fg">{formatMetric(claim.metric, claim.target)} {claim.metric?.toUpperCase()}</span>
      </div>
      <p className="mb-3 mt-0 text-[13px] leading-relaxed text-fg-4">{source.extracted ? 'Target quotation extracted from the paper.' : 'Recorded target source. Automatic target verification did not complete; this does not confirm the target or its table column.'}</p>
      {source.quote ? <blockquote className="m-0 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border-l-2 border-tone-amber-solid/50 bg-tone-amber-solid/5 px-4 py-3 font-mono text-[12px] leading-relaxed text-tone-amber-fg/90">“{source.quote}”</blockquote> : <p className="text-sm text-muted">No target quotation was recorded. Read the original source and the limitations of this check.</p>}
      <p className="mb-0 mt-3 text-[12px] leading-relaxed text-fg-4">
        {source.table && <>Recorded location: {source.table}. </>}
        {!source.table && source.recordedLocation && <>{source.recordedLocation}. </>}
        {source.columnVerified ? 'The table column was checked automatically.' : 'The table column was not verified automatically. Compare the method, dataset and metric in the original paper.'}
      </p>
      {source.url && <a className="mt-4 inline-block text-sm text-tone-blue underline underline-offset-4 hover:text-tone-blue-hover" href={source.url} target="_blank" rel="noreferrer">Open original source</a>}
    </div>
  )
}

// One line per seed run: which seed, whether it gave a valid result, what it measured. The selected branch of the tree is highlighted.
export function SeedMeasurements({ claimId, dossier, selected }: { claimId: string; dossier: Json; selected: Json | null }) {
  const attempts = measuredAttempts(dossier, claimId).sort((a, b) => a.seed - b.seed)
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const historyCount = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId).length - attempts.length
  return (
    <div className="min-w-0">
      <p className="m-0 mb-3 text-[12px] leading-relaxed text-muted">
        {attempts.length} selected measurements contribute to the verdict, one seed per sandbox branch. Single seeds may fall outside the band.
        {historyCount > 0 && <> {historyCount} other attempts remain in the branch tree above.</>}
      </p>
      <div className="max-h-[300px] overflow-auto border-y border-line">
        <table className="w-full text-left text-[12px]">
          <thead className="sticky top-0 bg-bg">
            <tr className="text-[10px] font-semibold uppercase tracking-wider text-muted">
              <th className="px-3 py-2.5 font-semibold">Seed</th>
              <th className="px-3 py-2.5 font-semibold">Result</th>
              <th className="px-3 py-2.5 text-right font-semibold">Measured</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {attempts.map((a, i) => (
              <tr key={i} className={selected?.branch_id === a.branch_id ? 'bg-tone-blue-solid/10' : ''}>
                <td className="px-3 py-2 font-mono text-fg-4">{a.seed}</td>
                <td className="px-3 py-2 text-fg-4">{a.measured != null ? 'Valid' : 'No valid result'}</td>
                <td className="px-3 py-2 text-right font-mono font-medium text-fg-2">{formatMetric(claim?.metric, a.measured)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function SeedChart({ claimId, dossier }: { claimId: string; dossier: Json }) {
  return (
    <div className="min-w-0">
      <h3 className={H3}>Each seed against the claim</h3>
      <AttemptChart dossier={dossier} claimId={claimId} />
    </div>
  )
}

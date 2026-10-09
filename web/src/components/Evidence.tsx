import { toPercent } from './ui'
import { AttemptChart } from './Gauge'
import type { Json } from '../data'

const ORDINAL = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']

// Which number of the quoted table row is our target? ("second number") Read from the quote itself; null when it cannot be told.
export function targetPosition(quote: string, target: number, metric: string): number | null {
  const nums = [...quote.replace(/(\d)\s+\.\s*(\d)/g, '$1.$2').matchAll(/\d+(?:\.\d+)?(?!\d)(?!s\b)/g)].map((m) => Number(m[0]))
  const want = metric === 'accuracy' ? target * 100 : target
  const i = nums.findIndex((n) => Math.abs(n - want) < (metric === 'accuracy' ? 0.051 : 0.0006))
  return i < 0 ? null : i
}

const H3 = 'm-0 mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted'

// The sentence of the paper that holds the target number, quoted from the page Tavily read.
export function TargetEvidence({ claim }: { claim: Json }) {
  const ev = claim?.target_evidence
  if (!ev) return <p className="m-0 text-sm italic text-muted">No verified target evidence found.</p>
  const i = targetPosition(String(ev.quote ?? ''), claim.target, claim.metric)
  return (
    <div className="min-w-0">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">Target value</span>
        <span className="font-mono text-lg font-semibold text-fg">{toPercent(claim.target)}</span>
      </div>
      <blockquote className="m-0 rounded-md border-l-2 border-tone-amber-solid/50 bg-tone-amber-solid/5 px-4 py-3 font-mono text-[12px] italic leading-relaxed text-tone-amber-fg/90">“{ev.quote}”</blockquote>
      <p className="mb-0 mt-3 text-[12px] leading-relaxed text-fg-4">
        Row from {ev.table || 'a table'} of the paper; {i == null ? 'our target is one of its numbers' : i < ORDINAL.length ? `the ${ORDINAL[i]} number is our target` : `our target is number ${i + 1}`}.
        {' '}<a className="text-tone-blue transition-colors hover:text-tone-blue-hover" href={ev.url} target="_blank" rel="noreferrer">Source page</a>
      </p>
    </div>
  )
}

// One line per seed run: which seed, whether it gave a valid result, what it measured. The selected branch of the tree is highlighted.
export function SeedMeasurements({ claimId, dossier, selected }: { claimId: string; dossier: Json; selected: Json | null }) {
  const attempts: Json[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId).sort((a: Json, b: Json) => a.seed - b.seed)
  const ok = attempts.filter((a) => a.measured != null)
  return (
    <div className="min-w-0">
      <p className="m-0 mb-3 text-[12px] leading-relaxed text-muted">
        {ok.length} of {attempts.length} runs gave a valid result, one seed per sandbox branch. The verdict uses the mean of all seeds; single seeds may fall outside the band.
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
                <td className="px-3 py-2 text-right font-mono font-medium text-fg-2">{toPercent(a.measured)}</td>
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

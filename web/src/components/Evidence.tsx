import { Badge, toPercent } from './ui'
import { ATTEMPT, de2en } from '../labels'
import type { Json } from '../data'

export function EvidenceSplit({ claimId, dossier,  selected }: { claimId: string; dossier: Json; events: any[]; selected: Json | null }) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const ev = claim?.target_evidence
  const attempts: Json[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId)
  const ok = attempts.filter((a) => a.measured != null)

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-5">
        <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted mb-3 border-b border-line/50 pb-2">Target Evidence (Claimed)</h3>
        {ev ? (
          <div className="bg-surface border border-line/50 rounded-xl p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between text-sm">
              <span className="text-fg-4 text-xs font-semibold uppercase tracking-wider">Target Value</span>
              <span className="font-mono text-fg text-lg font-semibold">{toPercent(claim.target)}</span>
            </div>
            <blockquote className="font-mono rounded-lg border-l-2 px-4 py-3 text-[12px] leading-relaxed border-tone-amber-solid/50 bg-tone-amber-solid/5 text-tone-amber-fg/90 italic">
              "{ev.quote}"
            </blockquote>
            <p className="mt-3 text-[11px] text-muted flex items-center gap-2">
              <span>{ev.table}</span>
              <span>•</span>
              <a className="text-tone-blue hover:text-tone-blue-hover transition-colors truncate max-w-[200px]" href={ev.url} target="_blank" rel="noreferrer">Source URL</a>
            </p>
          </div>
        ) : (
          <p className="text-sm text-muted italic">No verified target evidence found.</p>
        )}
      </div>

      <div className="space-y-5">
        <h3 className="text-[10px] font-semibold uppercase tracking-widest text-muted mb-3 border-b border-line/50 pb-2">Sandbox Measurements</h3>
        <div className="bg-surface border border-line/50 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-line/50 bg-panel/50">
            <div className="font-medium text-sm text-fg-2">Execution Results</div>
            <p className="text-[11px] text-muted mt-1">
              {ok.length} of {attempts.length} valid attempts across isolated seeds.
            </p>
          </div>
          <div className="max-h-[300px] overflow-auto">
            <table className="w-full text-[12px] text-left">
              <thead className="bg-panel/30 sticky top-0 backdrop-blur-sm">
                <tr>
                  <th className="py-3 px-4 border-b border-line/50 text-muted font-semibold uppercase tracking-wider text-[10px]">Seed</th>
                  <th className="py-3 px-4 border-b border-line/50 text-muted font-semibold uppercase tracking-wider text-[10px]">Status</th>
                  <th className="py-3 px-4 border-b border-line/50 text-muted font-semibold uppercase tracking-wider text-[10px] text-right">Measured</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/50">
                {attempts.map((a, i) => (
                  <tr key={i} className={`hover:bg-raised/50 transition-colors ${selected?.branch_id === a.branch_id ? 'bg-tone-blue-solid/10' : ''}`}>
                    <td className="py-3 px-4 font-mono text-sm text-fg-4">{a.seed}</td>
                    <td className="py-3 px-4">
                      <Badge tone={de2en(ATTEMPT, a.status).tone}>{de2en(ATTEMPT, a.status).label}</Badge>
                    </td>
                    <td className="py-3 px-4 font-mono text-sm text-right font-medium text-fg-2">{toPercent(a.measured)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

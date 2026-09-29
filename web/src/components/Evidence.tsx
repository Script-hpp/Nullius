import { ABLATION, ATTEMPT, de2en } from '../labels'
import type { Event, Json } from '../data'
import { Badge, fmt } from './ui'

const ROLE: Record<number, string> = {
  1: 'find material', 2: 'read claims', 3: 'collect objections', 4: 'setup details', 5: 'resolve error', 6: 'clarify deviation', 7: 'cite sources',
}

// Links: was Tavily fand. Rechts: was die Sandbox mass.
export function EvidenceSplit({ claimId, dossier, events, selected }: { claimId: string; dossier: Json; events: Event[]; selected: Json | null }) {
  const claim = (dossier.claims ?? []).find((c: Json) => c.id === claimId)
  const ev = claim?.target_evidence
  const calls = events.filter((e) => e.type === 'tavily_call')
  const details: Json[] = dossier.setup_details ?? []
  const attempts: Json[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId)
  const ab = dossier.ablations?.[claimId]
  const ok = attempts.filter((a) => a.measured != null)
  const bySource: Record<string, number> = {}
  for (const d of details) bySource[String(d.source).split('@')[0]] = (bySource[String(d.source).split('@')[0]] ?? 0) + 1

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--amber)' }}>Tavily found</h3>
        {ev ? (
          <div className="panel-2 p-4">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium">Target value</span>
              <span className="mono">{claim.target}</span>
              <Badge tone={ev.column_check === 'ok' ? 'green' : 'gray'} title="The table header is parsed and the position of the number in the row is checked deterministically.">
                column {ev.column_check === 'ok' ? 'verified against header' : 'not verifiable'}
              </Badge>
            </div>
            <blockquote className="mono rounded-md border-l-2 px-3 py-2 text-xs" style={{ borderColor: 'var(--amber)', background: 'rgba(245,184,65,0.06)' }}>
              “{ev.quote}”
            </blockquote>
            <p className="muted mt-2 text-xs">
              {ev.table} · <a className="underline" style={{ color: 'var(--blue)' }} href={ev.url} target="_blank" rel="noreferrer">{ev.url}</a> · version {ev.arxiv_version ?? '?'} · retrieved {String(ev.retrieved_at ?? '').slice(0, 10)}
            </p>
          </div>
        ) : (
          <p className="muted text-sm">No verified target from Tavily (target set manually).</p>
        )}
        {calls.length > 0 && (
          <div className="panel-2 p-4">
            <div className="mb-2 text-sm font-medium">Tavily calls in this run</div>
            <ul className="space-y-1.5 text-xs">
              {calls.map((c, i) => (
                <li key={i} className="flex items-start gap-2">
                  <Badge tone="amber">role {c.role}: {ROLE[c.role] ?? ''}</Badge>
                  <span className="muted mono break-all">{c.endpoint} · {String(c.target).slice(0, 70)} · {c.n_results} result(s)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {details.length > 0 ? (
          <div className="panel-2 p-4">
            <div className="mb-2 flex items-baseline justify-between text-sm">
              <span className="font-medium">Setup details ({details.length}, each with a verbatim quote)</span>
              <span className="muted text-xs">{Object.entries(bySource).map(([k, v]) => `${k}: ${v}`).join(' · ')}</span>
            </div>
            <ul className="max-h-72 space-y-2 overflow-auto pr-1 text-xs">
              {details.map((d, i) => (
                <li key={i}>
                  <div><b>{d.topic}</b>: <span className="mono">{String(d.value).slice(0, 120)}</span></div>
                  <div className="muted mono">“{String(d.quote).slice(0, 160)}” <span style={{ color: 'var(--blue)' }}>[{String(d.source).split('@')[0]}]</span></div>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="muted text-sm">Role 4 (setup details) was not used in this run.</p>
        )}
      </div>

      <div className="space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--green)' }}>Sandbox measured</h3>
        <div className="panel-2 p-4 text-sm">
          <div className="mb-2 font-medium">Independent harness</div>
          <p className="muted text-xs">
            The agent delivers predictions only. The harness is copied fresh into each branch, loads the test labels itself and computes the metric.
            {' '}{ok.length} of {attempts.length} attempts produced a valid measurement.
          </p>
          {dossier.isolation?.length > 0 && (
            <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs muted">{dossier.isolation.map((s: string, i: number) => <li key={i}>{s}</li>)}</ul>
          )}
        </div>
        {ab && (
          <div className="panel-2 p-4 text-sm">
            <div className="mb-2 flex items-center gap-2 font-medium">Graph ablation
              <Badge tone={ab.verdict === 'Graph genutzt' ? 'green' : ab.verdict === 'Graph nicht genutzt' ? 'red' : 'gray'}>{ABLATION[ab.verdict] ?? ab.verdict}</Badge>
            </div>
            <p className="muted text-xs">
              The same code runs again on the same data <b>without edges</b>. Accuracy with graph {fmt(ab.accuracy_full)} → without edges {fmt(ab.accuracy_ablated)}.
              If accuracy does not drop, the graph was never used.
              {ab.detail ? ` (${String(ab.detail).slice(0, 90)})` : ''}
            </p>
          </div>
        )}
        <div className="panel-2 max-h-80 overflow-auto p-4">
          <div className="mb-2 text-sm font-medium">Attempts</div>
          <table className="w-full text-xs">
            <thead className="muted"><tr><th className="pb-1 text-left">seed</th><th className="text-left">status</th><th className="text-right">measured</th><th className="text-right">code</th></tr></thead>
            <tbody>
              {attempts.map((a, i) => (
                <tr key={i} style={{ background: selected?.branch_id === a.branch_id ? 'rgba(90,176,255,0.10)' : undefined }}>
                  <td className="py-0.5">{a.seed}</td>
                  <td><Badge tone={de2en(ATTEMPT, a.status).tone}>{de2en(ATTEMPT, a.status).label}</Badge></td>
                  <td className="mono text-right">{fmt(a.measured)}</td>
                  <td className="mono muted text-right">{String(a.code_hash).slice(0, 7)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selected && (
          <div className="panel-2 p-4 text-xs">
            <div className="mb-1 text-sm font-medium">Branch <span className="mono">{selected.branch_id}</span></div>
            <div className="muted">seed {selected.seed} · {de2en(ATTEMPT, selected.status).label} · measured {fmt(selected.measured)} · cost so far ${fmt(selected.cost_usd, 4)}</div>
            {selected.error_summary && <pre className="mono mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-md p-2" style={{ background: 'rgba(255,107,107,0.08)' }}>{selected.error_summary}</pre>}
          </div>
        )}
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, CircleDashed, Loader2 } from 'lucide-react'
import { Badge, Section } from './ui'
import { Replay } from './Replay'
import { DATA_CHANGED, PHASES, deriveProgress, lastFinishedRun, useLiveRun } from '../live'
import { catalogLabel, isPreliminary, PRELIMINARY } from '../labels'
import { loadIndex, type RunRef } from '../data'
import { useCatalog } from '../catalog'
import { PreliminaryChip } from './CatalogBits'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'

const STATUS_TEXT = { connecting: 'Connecting', running: 'Running', finished: 'Finished', error: 'Failed', lost: 'Connection lost', unknown: 'Not found' } as const

export function LiveRun({ jobId }: { jobId: string }) {
  const { events, status, runId, error, verdict } = useLiveRun(jobId)
  const catalog = useCatalog()
  const [runs, setRuns] = useState<RunRef[]>([])
  useEffect(() => { loadIndex().then(setRuns) }, [status])

  const queued = events.find((e) => e.type === 'queued')
  const entry = catalog?.find((c) => c.id === queued?.catalog_id)
  const p = deriveProgress(events)
  const fallback = lastFinishedRun(entry, runs)
  const shownStatus = status === 'running' && !events.some((e) => e.type === 'started') ? 'connecting' : status
  const r = verdict ? catalogLabel(String(verdict.urteil), String(verdict.gesamtaussage)) : null

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <div className="pt-2 mb-6">
        <a href="#catalog" className="inline-flex items-center gap-1.5 text-muted text-xs font-medium uppercase tracking-wider hover:text-fg-2 transition-colors">
          <ArrowRight className="w-3.5 h-3.5 rotate-180" /> Back to catalog
        </a>
      </div>

      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-3 text-muted text-xs font-mono mb-2">
          <span>live reproduction</span>
          {queued && <span>• {queued.quick ? 'quick check, 3 seeds' : 'full run'}</span>}
          <span className={`inline-flex items-center gap-1.5 font-sans font-semibold uppercase tracking-wider ${status === 'error' || status === 'lost' || status === 'unknown' ? 'text-tone-red' : status === 'finished' ? 'text-tone-green' : 'text-fg-3'}`}>
            {status === 'finished' ? <CheckCircle2 className="w-3.5 h-3.5" /> : status === 'error' || status === 'lost' || status === 'unknown' ? <AlertCircle className="w-3.5 h-3.5" /> : <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {STATUS_TEXT[shownStatus]}
          </span>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-fg">{entry?.title ?? 'Reproduction run'}</h1>
        {entry && <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-fg-4">
          The loop re-runs the claim for {entry.claim.method} in an isolated sandbox. {status === 'running' || status === 'connecting' ? 'Everything below is what the agent and the harness are doing right now.' : 'Below is what the agent and the harness did.'}
        </p>}
      </div>

      <Section title="Progress" hint={status === 'finished' ? 'done' : 'updates live'} className="mb-6">
        <ol className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4 lg:grid-cols-8">
          {PHASES.map((ph) => {
            const done = p.done.includes(ph)
            const active = !done && p.phase === ph && status !== 'error' && status !== 'finished'
            return (
              <li key={ph} className={`flex items-center gap-2 text-[12px] ${done ? 'text-fg-3' : active ? 'text-fg font-medium' : 'text-subtle'}`}>
                {done ? <CheckCircle2 className="w-4 h-4 shrink-0 text-tone-green" /> : active ? <Loader2 className="w-4 h-4 shrink-0 animate-spin" /> : <CircleDashed className="w-4 h-4 shrink-0" />}
                {ph === 'Seeds' && p.seedsTotal ? `Seeds ${p.seedsDone}/${p.seedsTotal}` : ph}
              </li>
            )
          })}
        </ol>
        <div className="mt-5 flex flex-wrap gap-x-8 gap-y-2 border-t border-line/50 pt-4 text-[12px] text-muted">
          <span>Seeds <span className="font-mono text-fg-3">{p.seedsDone}{p.seedsTotal ? ` of ${p.seedsTotal}` : ''}</span></span>
          <span>Model cost (net) <span className="font-mono text-fg-3">${p.costNet.toFixed(3)}</span></span>
          {p.repairs > 0 && <span>Graph repairs <span className="font-mono text-fg-3">{p.repairs}</span></span>}
        </div>
      </Section>

      {status === 'finished' && verdict && r && (
        <div className="mb-6 rounded-xl border border-line bg-panel/50 p-5">
          <div className={label}>Result of this run</div>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
            <Badge tone={r.tone}>{r.label}</Badge>
            {isPreliminary(String(verdict.gesamtaussage)) && <PreliminaryChip />}
            <span className="font-mono text-[13px] text-fg-3">
              measured {typeof verdict.measured_mean === 'number' ? (entry?.claim.metric === 'accuracy' ? `${(verdict.measured_mean * 100).toFixed(1)}%` : verdict.measured_mean.toFixed(3)) : '–'}
            </span>
          </div>
          <p className="mt-2 text-[12px] leading-relaxed text-muted">
            {isPreliminary(String(verdict.gesamtaussage)) ? `${PRELIMINARY}. ` : ''}This is a quick check with our setup, not a finding about the paper. The curated result in the catalog is unchanged.
          </p>
          {runId && <a href={`#/run/${runId}`} className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-fg hover:text-fg-max">View evidence<ArrowRight className="w-4 h-4" /></a>}
        </div>
      )}

      {(status === 'error' || status === 'lost' || status === 'unknown') && (
        <div className="mb-6 rounded-xl border border-line bg-panel/50 p-5">
          <div className="flex items-start gap-3">
            <AlertCircle className="mt-0.5 w-4 h-4 shrink-0 text-tone-red" />
            <div className="min-w-0 text-[13px] leading-relaxed">
              <p className="font-medium text-fg">
                {status === 'error' ? 'This run did not finish.' : status === 'lost' ? 'The connection to the live backend was lost.' : 'This live run is no longer available.'}
              </p>
              {error && <p className="mt-1 break-words font-mono text-[12px] text-muted">{error}</p>}
              <p className="mt-2 text-muted">
                The sandbox or the model endpoint can be temporarily unstable. {fallback ? 'The last finished run is available as a replay.' : 'There is no finished run for this entry yet.'}
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
                {fallback && <a href={`#/run/${fallback}?tab=log`} className="inline-flex items-center gap-1.5 text-fg hover:text-fg-max">Watch the last finished run<ArrowRight className="w-4 h-4" /></a>}
                {status === 'lost' && <button onClick={() => { window.dispatchEvent(new Event(DATA_CHANGED)); location.reload() }} className="text-fg-3 underline underline-offset-4 cursor-pointer hover:text-fg">Reconnect</button>}
                <a href="#catalog" className="text-fg-3 underline underline-offset-4 hover:text-fg">Back to catalog</a>
              </div>
            </div>
          </div>
        </div>
      )}

      <Replay events={events} live ended={status === 'finished' || status === 'error' || status === 'lost' || status === 'unknown'} />
    </main>
  )
}

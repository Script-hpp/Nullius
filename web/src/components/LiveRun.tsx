import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, Loader2 } from 'lucide-react'
import { RunTimeline } from './run/RunTimeline'
import { StepGraph } from './run/StepGraph'
import { useLiveRun, lastFinishedRun, DATA_CHANGED } from '../live'
import { translateClaim } from '../labels'
import { loadIndex, type RunRef } from '../data'
import { useCatalog } from '../catalog'

const STATUS_TEXT = { connecting: 'Connecting', running: 'Running', finished: 'Finished', error: 'Failed', lost: 'Connection lost', unknown: 'Not found' } as const

export function LiveRun({ jobId }: { jobId: string }) {
  const { events, status, runId, error, lastReceived } = useLiveRun(jobId)
  const catalog = useCatalog()
  const [runs, setRuns] = useState<RunRef[]>([])
  useEffect(() => { loadIndex().then(setRuns) }, [status])

  const queued = events.find((e) => e.type === 'queued')
  const entry = catalog?.find((c) => c.id === queued?.catalog_id)
  const claimText = useMemo(() => {
    const cs = events.find((e) => e.type === 'claim_start')
    return cs?.text ? translateClaim(String(cs.text)) : null
  }, [events])
  const fallback = lastFinishedRun(entry, runs)
  const ended = status === 'finished' || status === 'error' || status === 'lost' || status === 'unknown'
  const shownStatus = status === 'running' && !events.some((e) => e.type === 'started') ? 'connecting' : status
  const bad = status === 'error' || status === 'lost' || status === 'unknown'

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-24">
      <div className="pt-2 mb-6">
        <a href="#catalog" className="inline-flex items-center gap-1.5 text-muted text-xs font-medium uppercase tracking-wider hover:text-fg-2 transition-colors">
          <ArrowRight className="w-3.5 h-3.5 rotate-180" /> Back to catalog
        </a>
      </div>

      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-3 text-muted text-xs font-mono mb-2">
          <span>live reproduction</span>
          {queued && <span>• {queued.quick ? 'quick check, 3 seeds' : 'full run'}</span>}
          <span className={`inline-flex items-center gap-1.5 font-sans font-semibold uppercase tracking-wider ${bad ? 'text-tone-red' : status === 'finished' ? 'text-tone-green' : 'text-fg-3'}`}>
            {status === 'finished' ? <CheckCircle2 className="w-3.5 h-3.5" /> : bad ? <AlertCircle className="w-3.5 h-3.5" /> : <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {STATUS_TEXT[shownStatus]}
          </span>
        </div>
        <h1 className="min-h-8 text-2xl font-semibold tracking-tight text-fg">{entry?.title ?? 'Reproduction run'}</h1>
        <p className="mt-2 min-h-5 max-w-3xl text-[13px] leading-relaxed text-fg-4">
          {claimText ? <>Claim: {claimText}. </> : null}
          {ended ? 'Below is what the agent and the harness did.' : 'Watch the agent re-run this claim in an isolated sandbox. Each step says what happens and why.'}
        </p>
      </div>

      {bad && (
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

      <section className="mb-10" aria-label="Step graph">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-muted">Step graph</h2>
        <StepGraph events={events} running={!ended} metricHint={entry?.claim.metric} />
      </section>

      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-muted">Log</h2>
      <RunTimeline events={events} mode="live" running={!ended} lastReceived={lastReceived} evidenceRunId={runId} metricHint={entry?.claim.metric} />
    </main>
  )
}

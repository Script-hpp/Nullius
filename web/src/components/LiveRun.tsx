import { useEffect, useMemo, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, Loader2 } from 'lucide-react'
import { RunTimeline } from './run/RunTimeline'
import { NextStep, PageIntro } from './page'
import { StepStrip, buildSteps, type StepId } from './run/StepStrip'
import { StepGraph } from './run/StepGraph'
import { useLiveRun, lastFinishedRun, DATA_CHANGED } from '../live'
import { translateClaim } from '../labels'
import { loadIndex, type RunRef } from '../data'
import { useCatalog } from '../catalog'
import { PaperHead } from './PaperHead'
import { useHead } from '../useHead'
import { headFromSummary } from '../paperhead'

const STATUS_TEXT = { connecting: 'Connecting', running: 'Running', finished: 'Finished', error: 'Failed', lost: 'Connection lost', unknown: 'Not found' } as const

export function LiveRun({ jobId }: { jobId: string }) {
  const { events, status, runId, error, lastReceived } = useLiveRun(jobId)
  const catalog = useCatalog()
  const [runs, setRuns] = useState<RunRef[]>([])
  useEffect(() => { loadIndex().then(setRuns) }, [status])

  const queued = events.find((e) => e.type === 'queued')
  const entry = catalog?.find((c) => c.id === queued?.catalog_id)
  const stored = useHead(entry?.id)
  const head = headFromSummary(events.find((e) => e.type === 'paper_summary')) ?? stored   // the head the run wrote itself, as soon as its event is there
  const seedsN = Number(events.find((e) => e.type === 'run_start')?.seeds)   // the seed count of this run, from the job itself; unknown before run_start
  const kind = queued ? (queued.quick ? 'quick check' : 'full run') + (Number.isFinite(seedsN) && seedsN > 0 ? `, ${seedsN} seeds` : '') : ''
  const claimText = useMemo(() => {
    const cs = events.find((e) => e.type === 'claim_start')
    return cs?.text ? translateClaim(String(cs.text)) : null
  }, [events])
  const fallback = lastFinishedRun(entry, runs)
  const ended = status === 'finished' || status === 'error' || status === 'lost' || status === 'unknown'
  const shownStatus = status === 'running' && !events.some((e) => e.type === 'started') ? 'connecting' : status
  const bad = status === 'error' || status === 'lost' || status === 'unknown'

  const steps = buildSteps(events, null, { running: !ended, metric: entry?.claim.metric })
  const PHASE_OF: Partial<Record<StepId, string>> = { paper: 'live-top', claim: 'phase-claim', sources: 'phase-setup', plan: 'phase-plan', code: 'phase-code', sandbox: 'phase-sandbox', check: 'phase-seeds', verdict: 'phase-verdict', community: 'phase-community' }
  const onStep = (id: StepId) => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    document.getElementById(PHASE_OF[id] ?? 'live-top')?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
  }

  return (
    <main id="live-top" className="mx-auto max-w-6xl px-4 pb-24 md:px-8">
      <div className="pt-6">
        <a href="#catalog" className="inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted no-underline transition-colors hover:text-fg-2">
          <ArrowRight className="h-3.5 w-3.5 rotate-180" /> Back to catalog
        </a>
      </div>
      <PageIntro kicker={`Live check${kind ? ` · ${kind}` : ''} · ${STATUS_TEXT[shownStatus]}`} title={entry?.title ?? 'Reproduction run'} head={<PaperHead head={head} />}
        what="You are watching one check as it happens. Our agent rebuilds the paper's experiment in isolated sandboxes while our harness measures the result."
        look="The nine steps below fill in as they finish. Open the running step to see what happens in it and why it matters.">
        <p className="mb-0 mt-8 flex items-center gap-3 text-[13px] text-fg-4">
          <span className={`inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider ${bad ? 'text-tone-red' : status === 'finished' ? 'text-tone-green' : 'text-fg-3'}`}>
            {status === 'finished' ? <CheckCircle2 className="h-3.5 w-3.5" /> : bad ? <AlertCircle className="h-3.5 w-3.5" /> : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {STATUS_TEXT[shownStatus]}
          </span>
          {claimText ? <span>Claim: {claimText}</span> : null}
        </p>
      </PageIntro>

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

      <div className="mb-14"><StepStrip steps={steps} onStep={onStep} /></div>

      <section id="step-graph" className="mb-12 scroll-mt-4" aria-label="Step graph">
        <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-muted">Step graph</h2>
        <StepGraph events={events} running={!ended} metricHint={entry?.claim.metric} />
      </section>

      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-muted">Log</h2>
      <RunTimeline events={events} mode="live" running={!ended} lastReceived={lastReceived} evidenceRunId={runId} metricHint={entry?.claim.metric} />
      {status === 'finished' && runId
        ? <NextStep title="Read the certificate" text="The check is finished. Open the certificate: the verdict in one sentence, claimed against measured, and why the number can be trusted." label="Open the certificate" href={`#/run/${runId}`} />
        : bad
          ? <NextStep title="Try another entry" text="This run did not finish. The catalog lists the checks that did, each with its evidence." label="Back to the catalog" href="#catalog" />
          : <NextStep title="When it finishes" text="The certificate of this check appears here as soon as the verdict is in. Until then, the catalog lists finished checks you can read." label="Open the catalog" href="#catalog" />}
    </main>
  )
}

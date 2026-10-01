import { useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { ResultBand } from './ResultChart'
import { buildSteps, type Step } from './run/StepStrip'
import { TIMELAPSE_MS, countAt, durationText, eventsForClaim, partialDossier } from '../timelapse'
import { TONE_VAR, type Tone } from '../labels'
import type { Event, Json } from '../data'

// "Play this check in 30 seconds": the finished run played back in a fixed 30 s. Step bar fills, sandbox branches appear, measurements fall
// into the tolerance band, then the verdict. Same data as the replay; only the clock is compressed.
export function Timelapse({ events, dossier, claim, tone, resultLabel, fmt, fmtSpread, autoplay, hasPlan }: {
  events: Event[]; dossier: Json; claim: Json; tone: Tone; resultLabel: string; fmt: (x: number) => string; fmtSpread: (x: number) => string
  autoplay: boolean; hasPlan: boolean
}) {   // autoplay: start at once when the panel opens
  const claimEvents = useMemo(() => eventsForClaim(events, claim.id), [events, claim.id])
  const total = claimEvents.length
  const [elapsed, setElapsed] = useState(0)
  const [playing, setPlaying] = useState(autoplay)
  const raf = useRef(0)
  const last = useRef(0)
  const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (!playing) return
    last.current = performance.now()
    const tick = (now: number) => {
      const dt = now - last.current
      last.current = now
      let done = false
      setElapsed((e) => { const n = e + dt; if (n >= TIMELAPSE_MS) { done = true; return TIMELAPSE_MS } return n })
      if (done) setPlaying(false)
      else raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing])

  const count = elapsed <= 0 && !playing ? 0 : countAt(elapsed, total)
  const seen = useMemo(() => claimEvents.slice(0, count), [claimEvents, count])
  const atEnd = count >= total && total > 0
  const part = useMemo(() => partialDossier(dossier, claim.id, seen, atEnd), [dossier, claim.id, seen, atEnd])
  const steps: Step[] = useMemo(() => buildSteps(seen, part, { running: !atEnd, hasPlan, metric: claim.metric }), [seen, part, atEnd, hasPlan, claim.metric])
  const attempts = seen.filter((e) => e.type === 'attempt')
  const verdictShown = (part.verdicts ?? []).length > 0
  const took = durationText(events)
  const doneSteps = steps.filter((s) => s.status === 'done').length

  const btn = 'inline-flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium'
  const start = () => { setElapsed(0); setPlaying(true) }
  return (
    <section id="sec-timelapse" className="scroll-mt-4 min-w-0 rounded-xl border border-line bg-surface p-6 md:p-8" aria-label="This check in 30 seconds">
      <div className="flex flex-wrap items-center gap-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">This check in 30 seconds</div>
        {took && <span className="text-[12px] text-muted">The real run took {took}.</span>}
        <div className="ml-auto flex items-center gap-2">
          <button type="button" onClick={() => (atEnd || elapsed === 0 ? start() : setPlaying((p) => !p))} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
            {playing ? <><Pause className="h-3.5 w-3.5" />Pause</> : <><Play className="h-3.5 w-3.5" />{atEnd ? 'Play again' : elapsed === 0 ? 'Play' : 'Resume'}</>}
          </button>
          <button type="button" onClick={start} className={`${btn} border-line text-fg-3 hover:bg-raised`}><RotateCcw className="h-3.5 w-3.5" />Restart</button>
        </div>
      </div>

      <div className="mt-6" role="progressbar" aria-label="Steps done" aria-valuemin={0} aria-valuemax={steps.length} aria-valuenow={doneSteps}>
        <div className="flex gap-1">
          {steps.map((s) => (
            <div key={s.id} className="min-w-0 flex-1">
              <div className="h-1.5 rounded-full bg-raised-2" style={{ background: s.status === 'done' ? 'var(--c-fg-4)' : s.status === 'running' ? 'var(--c-line-strong)' : undefined, transition: reduce ? undefined : 'background 300ms' }} />
              <div className={`mt-2 hidden truncate text-[10px] uppercase tracking-wider md:block ${s.status === 'done' ? 'text-fg-3' : 'text-muted'}`}>{s.title}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-[12px] text-muted md:hidden">{doneSteps} of {steps.length} steps done</div>
      </div>

      <div className="mt-6 min-w-0">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted">Sandbox branches: {attempts.length}</div>
        <div className="mt-2 flex min-h-[1.25rem] flex-wrap gap-1.5" aria-hidden>
          {attempts.map((a, i) => {
            const ok = typeof a.measured === 'number'
            return <span key={i} title={ok ? 'valid run' : 'failed run'} className="h-3 w-3 rounded-[3px]" style={{ background: ok ? TONE_VAR.green : TONE_VAR.red, opacity: ok ? 1 : 0.7, animation: reduce ? undefined : 'tl-pop 250ms ease-out' }} />
          })}
        </div>
      </div>

      <div className="mt-6 min-w-0">
        <ResultBand dossier={part} claimId={claim.id} tone={verdictShown ? tone : 'gray'} fmt={fmt} fmtSpread={fmtSpread} />
      </div>
      <div className="mt-4 min-h-[2.5rem]">
        {verdictShown && <div className="font-grotesk text-2xl font-semibold tracking-[-0.03em]" style={{ color: TONE_VAR[tone] }}>{resultLabel}</div>}
      </div>
    </section>
  )
}

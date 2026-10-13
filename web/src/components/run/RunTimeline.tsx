import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ArrowRight, Box, CheckCircle2, ChevronDown, CircleDashed, Cpu, Loader2, Search, ShieldCheck, ArrowDown, Info } from 'lucide-react'
import { AttemptChart } from '../Gauge'
import { Hint } from '../Hint'
import { Badge, Section } from '../ui'
import { PreliminaryChip } from '../CatalogBits'
import { PythonCode } from './PythonCode'
import { PlanText } from './PlanText'
import { TavilyMark } from '../TavilyMark'
import { ACTOR_LABEL, buildRunModel, fmtClock, fmtValue, planMapping, type Actor, type Expand, type Phase, type PhaseId, type Row } from './phases'
import { TONE_TEXT, catalogLabel, isPreliminary, resultHint, translateClaim, translateWarning, PRELIMINARY } from '../../labels'
import type { Event } from '../../data'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'
const ACTOR_ICON: Record<Actor, typeof Search> = { tavily: Search, nemotron: Cpu, sandbox: Box, harness: ShieldCheck, system: Info }
const ACTOR_HINT: Record<Actor, string> = {
  tavily: 'Tavily is the researcher: it fetches pages from the web (paper, README) for the agent.',
  nemotron: 'Nemotron is the brain: NVIDIA language models that plan, write code and read text. Different roles use different model sizes.',
  sandbox: 'The sandbox is the lab: an isolated machine where the agent code runs. Every seed runs in its own branch of the same snapshot.',
  harness: 'The harness loads the targets separately after training and computes the score from the saved predictions.',
  system: 'Infrastructure: retries and waiting.',
}
const TERM_HINT = {
  seed: 'A seed fixes the random numbers of one training run. Different seeds show how much the result varies by chance.',
  tolerance: 'The allowed distance between the measured mean and the claimed value. It is fixed before the run.',
  branch: 'A sandbox branch is a copy of the prepared environment. Seeds run in separate copies and cannot influence each other.',
  graphCheck: 'We remove all graph edges and run the same code again. If accuracy does not drop, the code ignores the graph.',
}

function ActorChip({ row }: { row: Row }) {
  if (row.actor === 'tavily') return <span className="inline-flex w-fit shrink-0 items-center justify-center self-start rounded-md border border-line px-1.5 py-0.5"><TavilyMark kind={row.tavily} className="text-[10px] uppercase tracking-wider" /></span>
  const Icon = ACTOR_ICON[row.actor]
  return (
    <span className="inline-flex shrink-0 flex-col items-start gap-0.5 self-start">
      <span className="inline-flex w-fit items-center justify-center gap-1.5 whitespace-nowrap rounded-md border border-line px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-fg-4">
        <Icon className="w-3 h-3" />{ACTOR_LABEL[row.actor]}
      </span>
      {row.role && <span className="px-1 font-mono text-[10px] leading-tight text-muted break-all" title={row.model}>{row.role}{row.model ? <span className="hidden sm:inline"> · {row.model}</span> : null}</span>}
    </span>
  )
}

function useCode(hash: string, inline: string | undefined, loader?: (hash: string) => Promise<string | null>) {
  const [loaded, setLoaded] = useState<{ hash: string; text: string | null } | null>(null)
  useEffect(() => {
    if (inline || !loader) return
    let cancelled = false
    loader(hash).then((t) => { if (!cancelled) setLoaded({ hash, text: t }) })
    return () => { cancelled = true }
  }, [hash, inline, loader])
  return inline ?? (loaded?.hash === hash ? loaded.text : null)
}

export function ExpandView({ ex, loader, planFull }: { ex: Expand; loader?: (hash: string) => Promise<string | null>; planFull?: string | null }) {
  const [all, setAll] = useState(false)
  if (ex.kind === 'quote') return <blockquote className="mt-2 border-l-2 border-line-strong pl-3 font-mono text-[11px] leading-relaxed text-muted break-words">{ex.quote}</blockquote>
  if (ex.kind === 'error') return <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-surface p-2.5 font-mono text-[11px] text-muted">{ex.text}</pre>
  if (ex.kind === 'setup') {
    return (
      <ul className="mt-2 space-y-1.5">
        {ex.items.map((d, i) => (
          <li key={i} className="text-[12px] leading-relaxed text-fg-3">
            <span className="text-fg-2">{d.topic}</span>: {d.value} <span className="text-muted">({d.source})</span>
            {d.quote && <span className="block border-l border-line pl-2 font-mono text-[10.5px] text-subtle break-words">“{d.quote}”</span>}
          </li>
        ))}
      </ul>
    )
  }
  if (ex.kind === 'plan') {
    const text = planFull ?? ex.text
    const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
    const visible = all ? lines : lines.slice(0, 6)
    return (
      <div className="mt-2">
        <PlanText lines={visible} />
        {lines.length > 6 && <button onClick={() => setAll(!all)} className="mt-1.5 text-[11px] text-muted hover:text-fg-3 cursor-pointer">{all ? 'Show less' : `Show all ${lines.length} lines`}</button>}
      </div>
    )
  }
  return <CodeView ex={ex} loader={loader} />
}

function CodeView({ ex, loader }: { ex: Extract<Expand, { kind: 'code' }>; loader?: (hash: string) => Promise<string | null> }) {
  const code = useCode(ex.hash, ex.inline, loader)
  if (code == null) return <p className="mt-2 text-[11px] text-muted">Code version {ex.version} ({ex.hash}) is not stored for this run.</p>
  const mapping = planMapping(code)
  return (
    <div className="mt-2 space-y-2">
      {mapping && (
        <div className="rounded-lg border border-line bg-surface p-3">
          <p className={label}>What the agent says it implemented</p>
          <ul className="mt-1.5 space-y-0.5 text-[12px] leading-relaxed text-fg-3">{mapping.map((l, i) => <li key={i} className="break-words">{l}</li>)}</ul>
          <p className="mt-1.5 text-[10.5px] text-muted">The agent&apos;s own statement from the top of its code. We do not check it.</p>
        </div>
      )}
      <PythonCode code={code} />
    </div>
  )
}

function RowView({ row, t0, loader, planFull, reduce }: { row: Row; t0: number | null; loader?: (hash: string) => Promise<string | null>; planFull?: string | null; reduce: boolean }) {
  const [open, setOpen] = useState(false)
  const [raw, setRaw] = useState(false)
  const Dot = TONE_TEXT[row.tone]
  const EXPAND_LABEL: Record<Expand['kind'], string> = { plan: 'the plan', code: 'the code', setup: 'the details', quote: 'the quote', error: 'the error' }
  return (
    <motion.li initial={reduce ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className="grid grid-cols-1 gap-x-3 gap-y-1.5 sm:grid-cols-[9rem_1fr_auto]">
      <ActorChip row={row} />
      <div className="min-w-0">
        <div className="flex items-start gap-2">
          <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-current ${Dot}`} />
          <p className="text-[13px] leading-snug text-fg-2">{row.title}</p>
        </div>
        {row.detail && <p className="ml-3.5 mt-0.5 text-[12px] leading-relaxed text-muted break-words">{row.detail}</p>}
        <div className="ml-3.5 mt-1 flex flex-wrap gap-x-4 gap-y-1">
          {row.expand && <button onClick={() => setOpen(!open)} className="inline-flex items-center gap-1 text-[11px] text-fg-4 hover:text-fg cursor-pointer"><ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />{open ? 'Hide' : 'Show'} {EXPAND_LABEL[row.expand.kind]}</button>}
          {row.raw.length > 0 && <button onClick={() => setRaw(!raw)} className="text-[11px] text-subtle hover:text-fg-3 cursor-pointer">{raw ? 'Hide raw event' : 'Details'}</button>}
        </div>
        <div className="ml-3.5">
          {open && row.expand && <ExpandView ex={row.expand} loader={loader} planFull={planFull} />}
          {raw && <pre className="mt-2 max-h-48 overflow-auto rounded-lg border border-line bg-surface p-2.5 font-mono text-[10.5px] text-muted">{JSON.stringify(row.raw.length === 1 ? row.raw[0] : row.raw, (_k, v) => (typeof v === 'string' && v.length > 400 ? v.slice(0, 400) + '…' : v), 1)}</pre>}
        </div>
      </div>
      {t0 != null && <span className="hidden sm:block font-mono text-[10px] text-subtle pt-1">+{fmtClock(row.ts - t0)}</span>}
    </motion.li>
  )
}

function PhaseCard({ phase, index, open, onToggle, t0, loader, planFull, waiting, reduce, hintFor }: {
  phase: Phase; index: number; open: boolean; onToggle: () => void; t0: number | null; loader?: (hash: string) => Promise<string | null>
  planFull?: string | null; waiting: string | null; reduce: boolean; hintFor?: ReactNode
}) {
  const active = phase.state === 'active'
  return (
    <section id={`phase-${phase.id}`} className={`rounded-xl border bg-surface scroll-mt-4 ${active ? 'border-line-stronger' : 'border-line'} ${phase.state === 'pending' ? 'opacity-75' : ''}`}>
      <button onClick={onToggle} disabled={phase.state === 'pending'} aria-expanded={open}
        className="flex w-full items-start gap-3 p-4 text-left cursor-pointer disabled:cursor-default">
        <span className="mt-0.5 shrink-0">
          {phase.state === 'done' ? <CheckCircle2 className="w-4 h-4 text-tone-green" /> : active ? <Loader2 className="w-4 h-4 animate-spin text-fg-2" /> : <CircleDashed className="w-4 h-4 text-subtle" />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-sm font-medium text-fg"><span className="font-mono text-[11px] text-subtle mr-2">{index + 1}</span>{phase.title}</span>
            <span className="flex gap-1">{phase.actors.map((a) => {
              const Icon = ACTOR_ICON[a]
              if (a === 'tavily') return <span key={a} className="inline-flex rounded-md border border-line px-1.5 py-0.5"><TavilyMark kind={phase.rows.find((r) => r.actor === 'tavily')?.tavily} className="!text-[10px] uppercase tracking-wider !text-muted" title={ACTOR_HINT[a]} /></span>
              return <span key={a} title={ACTOR_HINT[a]} className="inline-flex items-center gap-1 rounded-md border border-line px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted"><Icon className="w-3 h-3" />{ACTOR_LABEL[a]}</span>
            })}</span>
            {hintFor}
          </span>
          {phase.state === 'done' && phase.summary && <span className="mt-1 block text-[12px] text-fg-4">{phase.summary}</span>}
          {phase.state === 'active' && <span className="mt-1 block text-[12px] text-fg-4">{phase.what}</span>}
          {phase.state === 'pending' && <span className="mt-1 block text-[12px] text-muted">{phase.what}</span>}
        </span>
        {phase.state !== 'pending' && <ChevronDown className={`mt-1 w-4 h-4 shrink-0 text-muted transition-transform ${open ? 'rotate-180' : ''}`} />}
      </button>
      <AnimatePresence initial={false}>
        {open && phase.state !== 'pending' && (
          <motion.div initial={reduce ? false : { opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.2 }} className="overflow-hidden">
            <div className="space-y-4 border-t border-line/60 px-4 pb-4 pt-3">
              <dl className="grid gap-2 text-[12px] leading-relaxed sm:grid-cols-2">
                <div><dt className={label}>What happens</dt><dd className="mt-0.5 text-fg-3">{phase.what}</dd></div>
                <div><dt className={label}>Why</dt><dd className="mt-0.5 text-fg-3">{phase.why}</dd></div>
              </dl>
              {phase.rows.length > 0 && <ul className="space-y-3.5">{phase.rows.map((r) => <RowView key={r.id} row={r} t0={t0} loader={loader} planFull={planFull} reduce={reduce} />)}</ul>}
              {active && waiting && (
                <div className="flex items-center gap-2 text-[12px] text-fg-4">
                  <motion.span animate={reduce ? undefined : { opacity: [1, 0.35, 1] }} transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }} className="inline-flex"><Loader2 className="w-3.5 h-3.5 animate-spin" /></motion.span>
                  <motion.span animate={reduce ? undefined : { opacity: [1, 0.55, 1] }} transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}>{waiting}</motion.span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [active])
  return now
}

export type RunTimelineProps = {
  events: Event[]
  mode: 'live' | 'replay'
  running: boolean
  lastReceived?: number                 // live: local time of the last event, for the clock between events
  embedded?: boolean                    // inside the evidence page: no measurements and no result card (the page shows them)
  codeLoader?: (hash: string) => Promise<string | null>
  planFull?: string | null
  evidenceRunId?: string | null         // live: link target after the run
  metricHint?: string | null            // metric of the catalog entry, known before claim_start arrives
}

export function RunTimeline({ events, mode, running, lastReceived, embedded = false, codeLoader, planFull, evidenceRunId, metricHint }: RunTimelineProps) {
  const reduce = !!useReducedMotion()
  const model = useMemo(() => buildRunModel(events, { running, metricHint }), [events, running, metricHint])
  const now = useNow(running && mode === 'live')
  const [manual, setManual] = useState<Partial<Record<PhaseId, boolean>>>({})
  const endRef = useRef<HTMLDivElement>(null)
  const [following, setFollowing] = useState(false)   // the page starts with the intro and the step strip: follow only once the reader is at the end
  const followRef = useRef(false)
  const setFollow = (v: boolean) => { followRef.current = v; setFollowing(v) }
  // Follow new events until the reader scrolls up (wheel, touch, keys or a scroll that moves up). Growing content never stops following.
  useEffect(() => {
    if (!endRef.current) return
    const io = new IntersectionObserver(([en]) => { if (en.isIntersecting) setFollow(true) }, { rootMargin: '0px 0px 120px 0px' })
    io.observe(endRef.current)
    let touchY = 0
    let lastTop = -1
    let lastHeight = 0
    const up = () => setFollow(false)
    const onWheel = (e: WheelEvent) => { if (e.deltaY < 0) up() }
    const onTouchStart = (e: TouchEvent) => { touchY = e.touches[0]?.clientY ?? 0 }
    const onTouchMove = (e: TouchEvent) => { if ((e.touches[0]?.clientY ?? 0) > touchY + 8) up() }
    const onKey = (e: KeyboardEvent) => { if (['ArrowUp', 'PageUp', 'Home'].includes(e.key)) up() }
    const onScroll = (e: globalThis.Event) => {
      const t = e.target as HTMLElement
      if (!(t instanceof HTMLElement)) return
      // a shrinking page (a phase card collapsing) also lowers scrollTop; that is not the reader scrolling up
      if (lastTop >= 0 && t.scrollHeight >= lastHeight && t.scrollTop < lastTop - 40) up()
      lastTop = t.scrollTop
      lastHeight = t.scrollHeight
    }
    window.addEventListener('wheel', onWheel, { passive: true })
    window.addEventListener('touchstart', onTouchStart, { passive: true })
    window.addEventListener('touchmove', onTouchMove, { passive: true })
    window.addEventListener('keydown', onKey)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      io.disconnect()
      window.removeEventListener('wheel', onWheel); window.removeEventListener('touchstart', onTouchStart); window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('keydown', onKey); window.removeEventListener('scroll', onScroll, true)
    }
  }, [])
  useEffect(() => {
    if (!running) return
    const go = () => { if (followRef.current) endRef.current?.scrollIntoView({ block: 'nearest', behavior: 'auto' }) }
    go()
    const t = window.setTimeout(go, 280)   // again after the phase card finished growing
    return () => window.clearTimeout(t)
  }, [events.length, running])

  const t0 = model.startTs
  const elapsed = t0 == null ? null : (model.endTs ?? t0) - t0 + (running && mode === 'live' && lastReceived ? Math.max(0, (now - lastReceived) / 1000) : 0)
  const m = model
  const attempts = m.attempts
  const dossier = useMemo(() => ({
    claims: [{ id: 'c', target: m.target, tolerance: m.tolerance ?? 0, metric: m.metric ?? 'accuracy' }],
    attempts: attempts.map((a) => ({ claim_id: 'c', seed: a.seed, measured: a.measured, status: m.target != null && Math.abs(a.measured - m.target) <= (m.tolerance ?? 0) ? 'stabil' : 'abweichung' })),
    baselines: {},
  }), [attempts, m.target, m.tolerance, m.metric])
  const delta = m.stats && m.target != null ? (m.stats.mean - m.target) * (m.metric === 'accuracy' ? 100 : 1) : null
  const verdict = m.verdict
  const label_ = verdict ? catalogLabel(String(verdict.urteil), String(verdict.gesamt)) : null

  const phaseHint: Partial<Record<PhaseId, ReactNode>> = {
    sandbox: <span className="inline-flex gap-0.5"><Hint text={TERM_HINT.branch} /><Hint text={TERM_HINT.graphCheck} /></span>,
    seeds: <Hint text={TERM_HINT.seed} />,
    verdict: <Hint text={TERM_HINT.tolerance} />,
  }

  return (
    <div>
      {!embedded && (
        <div className="mb-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-[12px] text-muted">
          <span>Runtime <span className="font-mono text-fg-3">{elapsed == null ? '–' : fmtClock(elapsed)}</span></span>
          {m.costNet != null && <span>Model cost (net) <span className="font-mono text-fg-3">${m.costNet.toFixed(3)}</span></span>}
          <span className="inline-flex items-center gap-2"><TavilyMark kind={m.tavilyKind ?? undefined} /><span className="font-mono text-fg-3">{m.tavilyCalls} {m.tavilyCalls === 1 ? 'call' : 'calls'}</span></span>
          {m.llmCalls > 0 && <span>Nemotron calls <span className="font-mono text-fg-3">{m.llmCalls}</span>{m.llmCached > 0 ? <span className="text-tone-amber"> ({m.llmCached} cached)</span> : <span> live</span>}</span>}
          {m.seedsTotal != null && <span>Seeds <span className="font-mono text-fg-3">{m.stats?.n ?? 0} of {m.seedsTotal}</span><Hint text={TERM_HINT.seed} /></span>}
        </div>
      )}

      <ol className="mb-5 flex flex-wrap gap-1.5" aria-label="Phases">
        {m.phases.map((p, i) => (
          <li key={p.id}>
            <a href={`#phase-${p.id}`} onClick={(e) => { e.preventDefault(); document.getElementById(`phase-${p.id}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' }) }}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11px] font-medium no-underline ${p.state === 'active' ? 'border-line-stronger text-fg' : p.state === 'done' ? 'border-line text-fg-3' : 'border-line/60 text-subtle'}`}>
              {p.state === 'done' ? <CheckCircle2 className="w-3.5 h-3.5 text-tone-green" /> : p.state === 'active' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <span className="font-mono text-[10px]">{i + 1}</span>}
              {p.short}
            </a>
          </li>
        ))}
      </ol>

      <div className={embedded ? '' : 'grid items-start gap-6 lg:grid-cols-3'}>
        <div className={`space-y-3 ${embedded ? '' : 'lg:col-span-2'}`}>
          {m.phases.map((p, i) => (
            <PhaseCard key={p.id} phase={p} index={i} open={manual[p.id] ?? p.state === 'active'} onToggle={() => setManual((s) => ({ ...s, [p.id]: !(s[p.id] ?? p.state === 'active') }))}
              t0={t0} loader={codeLoader} planFull={planFull} waiting={m.waiting} reduce={reduce} hintFor={phaseHint[p.id]} />
          ))}
        </div>

        {!embedded && (
          <div className="space-y-4 lg:sticky lg:top-4">
            <Section title="Measurements" hint={m.seedsTotal ? `${m.stats?.n ?? 0} of ${m.seedsTotal} seeds` : undefined}>
              {m.candidateSelectionPending && <p className="mb-3 text-[13px] leading-relaxed text-fg-3">Code candidates are being tested in parallel. Their scores are shown in the attempt history. The seed chart starts after candidate selection.</p>}
              {m.selectedCandidate != null && <p className="mb-3 text-[13px] text-fg-3">Selected candidate {m.selectedCandidate + 1}{!verdict ? ' · provisional measurements, no verdict yet' : ''}</p>}
              {m.measurementSelectionUnavailable && <p className="mb-3 text-[13px] leading-relaxed text-fg-3">The final seed selection cannot be reconstructed from this older event log. Open the full evidence for the recorded measurements.</p>}
              {m.firstMeasurement && m.target != null ? (
                <>
                  <AttemptChart dossier={dossier} claimId="c" />
                  {m.stats && (
                    <p className="mt-3 font-mono text-[12px] text-fg-3">
                      Mean {fmtValue(m.metric, m.stats.mean)} ± {fmtValue(m.metric, m.stats.std)}
                      {delta != null && <span className="text-muted"> · {delta >= 0 ? '+' : '−'}{Math.abs(delta).toFixed(m.metric === 'accuracy' ? 1 : 3)}{m.metric === 'accuracy' ? ' pp' : ''} vs. claimed</span>}
                    </p>
                  )}
                  <p className="mt-3 flex items-start gap-2 rounded-lg border border-line bg-surface p-3 text-[13px] leading-relaxed text-fg-3">
                    <ShieldCheck className="mt-0.5 w-3.5 h-3.5 shrink-0 text-muted" />
                    The agent writes predictions. Our harness loads the targets separately after training and computes the score. Method fidelity requires separate checks.
                  </p>
                </>
              ) : (
                !m.candidateSelectionPending && !m.measurementSelectionUnavailable && <p className="m-0 text-[14px] leading-relaxed text-fg-3">The first measurement appears here when seed 0 has finished. Each bar will be one seed; the band is the claimed value ± tolerance.</p>
              )}
            </Section>

            {verdict && label_ && (
              <div className="nx-card rounded-xl border border-line bg-panel/50 p-5">
                <div className={label}>Result</div>
                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-2">
                  <Badge tone={label_.tone} title={resultHint(String(verdict.urteil))}>{label_.label}</Badge>
                  {isPreliminary(String(verdict.gesamt)) && <PreliminaryChip />}
                </div>
                <p className="mt-2 text-[12px] leading-relaxed text-fg-3">
                  Measured {fmtValue(m.metric, Number(verdict.mean))} ± {fmtValue(m.metric, Number(verdict.std))} against the claimed {fmtValue(m.metric, m.target, 2)} (tolerance ±{fmtValue(m.metric, m.tolerance, m.metric === 'accuracy' ? 1 : 3)}).
                </p>
                <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
                  {isPreliminary(String(verdict.gesamt)) ? `${PRELIMINARY}. ` : ''}This is a measurement with our setup, not a finding about the paper.
                </p>
                {m.warnings.length > 0 && (
                  <ul className="mt-3 space-y-1 border-t border-line/60 pt-3">
                    {m.warnings.map((w, i) => <li key={i} className="text-[11px] leading-relaxed text-muted">{translateWarning(w)}</li>)}
                  </ul>
                )}
                {evidenceRunId && <a href={`#/run/${evidenceRunId}`} className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-fg hover:text-fg-max">Open full evidence<ArrowRight className="w-4 h-4" /></a>}
              </div>
            )}
          </div>
        )}
      </div>

      <div ref={endRef} className="h-px" />
      {!following && running && (
        <div className="pointer-events-none sticky bottom-6 z-10 mt-2 flex justify-center">
          <button onClick={() => { setFollow(true); endRef.current?.scrollIntoView({ block: 'end', behavior: reduce ? 'auto' : 'smooth' }) }}
            className="pointer-events-auto inline-flex items-center gap-1.5 rounded-full border border-line-strong bg-raised px-3.5 py-1.5 text-[12px] font-medium text-fg shadow-lg cursor-pointer">
            <ArrowDown className="w-3.5 h-3.5" />Jump to live
          </button>
        </div>
      )}
      {m.claimText && embedded && <p className="sr-only">{translateClaim(m.claimText)}</p>}
    </div>
  )
}

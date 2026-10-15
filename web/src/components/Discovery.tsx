import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Activity, CheckCircle2, Check, ChevronDown, FileText, Play, RotateCcw, Search, XCircle, AlertCircle, BookOpen, Code, Sparkles, PauseCircle, Loader2 } from 'lucide-react'
import { ExtArrow, ExtLink, ICON_MD, INLINE, Section } from './ui'
import { Hint } from './Hint'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, evidenceHref, useCatalog, type CatalogEntry } from '../catalog'
import { translateReason } from '../labels'
import { fetchJobInfo, startReproduction, useLiveStatus, useReproductionEligibility, type ReproductionEligibility } from '../live'
import { TavilyMark } from './TavilyMark'
import { isPaused, resumeText, startErrorText, statusFor } from '../livestatus'
import {
  deriveView, domainOf, formatMetric, humanReject, parseJsonl, parseLiveEvent, quoteCells, quoteExcerpt, shorten, stageLabel, benchmarkText, tickedRuntime, savedCheckForClaim,
  type DEvent, type DPaper, type DiscoveryView, type FeedCall,
} from '../discovery'
import { Glossed } from './Glossed'
import { NextStep, PageIntro } from './page'
import { useSummary } from '../summaries'

type Mode = 'idle' | 'live' | 'replay' | 'done' | 'failed' | 'saved'

const DATASETS = ['Cora', 'CiteSeer', 'ETTh1']
const DEFAULT_DATASETS = ['Cora', 'ETTh1']   // each ticked benchmark costs Tavily credits, so the default stays at two
const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'
const btn = 'inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed'

const PURPOSE_TITLE: Record<string, string> = { find_papers: 'Find papers', read_paper: 'Read paper', find_code: 'Find code', community: 'Community' }

const fmtDate = (ts: number) => new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
const fmtSec = (s: number) => `${Math.max(0, s).toFixed(1)}s`

const HINTS = {
  protocol: 'Protocol = how the paper split the data (and, for forecasting, input length and horizon). Numbers are only comparable when our re-run uses the same split.',
  column: "We check that the number sits in the proposed method's column of the results table, not in a baseline column.",
  unverified: 'This discovery run reads and selects a claim. It does not execute an experiment. Saved reproduction checks are shown separately.',
  cache: 'Read in an earlier run; no new API calls.',
}

function useTween() {
  const reduce = useReducedMotion()
  return reduce ? { duration: 0 } : { duration: 0.2, ease: 'easeOut' as const }
}

function FeedCard({ call, recorded }: { call: FeedCall; recorded: boolean }) {
  const [open, setOpen] = useState(false)
  const t = useTween()
  const Icon = call.endpoint === 'search' ? Search : FileText
  return (
    <motion.div layout="position" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={t}
      className="nx-card rounded-xl border border-line bg-panel/50">
      <button onClick={() => setOpen(!open)} className="w-full text-left p-4 cursor-pointer">
        <div className="flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-fg-3">
            <Icon className="w-3.5 h-3.5" />
            {PURPOSE_TITLE[call.purpose ?? ''] ?? call.endpoint}
            {call.purpose && <span className="text-subtle">{call.endpoint}</span>}
          </span>
          <span className="flex items-center gap-3 font-mono text-[11px] text-muted">
            <span>{call.hits} {call.hits === 1 ? 'hit' : 'hits'}</span>
            <span>{(call.durationMs / 1000).toFixed(2)}s</span>
            <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
          </span>
        </div>
        <p className="mt-2 font-mono text-[12px] leading-relaxed text-fg-4 break-all">
          {call.purpose === 'find_code' || call.purpose === 'community' ? <span className="text-subtle">Paper: </span> : null}
          {shorten(call.query, 110)}
        </p>
        <p className="mt-2"><TavilyMark kind={recorded ? 'recorded' : 'live'} /></p>
      </button>
      {open && call.topHits.length > 0 && (
        <ul className="border-t border-line/60 px-4 py-3 space-y-2">
          {call.topHits.map((h) => (
            <li key={h.url} className="text-[12px] leading-snug">
              <ExtLink href={h.url} className="text-fg-3 hover:text-fg-max no-underline">{h.title || h.url}</ExtLink>
              <div className="font-mono text-[10px] text-subtle">{domainOf(h.url)}</div>
            </li>
          ))}
        </ul>
      )}
    </motion.div>
  )
}

function QuoteView({ quote, target }: { quote?: string; target?: number }) {
  const [open, setOpen] = useState(false)
  const cells = useMemo(() => quoteCells(quote, target), [quote, target])
  const cellCls = 'whitespace-nowrap'
  if (!cells) {
    const [b, n, a] = quoteExcerpt(quote, target)
    return (
      <blockquote className="border-l-2 border-line pl-3 font-mono text-[11px] leading-relaxed text-muted break-words">
        {b}<span className="text-fg-strong font-semibold underline decoration-muted underline-offset-2">{n}</span>{a}
      </blockquote>
    )
  }
  const hi = <span className="text-fg-strong font-semibold underline decoration-muted underline-offset-2">{cells.cell[1]}</span>
  return (
    <div className="border-l-2 border-line pl-3 font-mono text-[11px] leading-relaxed text-muted">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {cells.head.map((c, i) => <span key={`h${i}`} className={`${cellCls} text-fg-4`}>{c}</span>)}
        {cells.hasGap && <span className="text-faint">…</span>}
        {cells.before.map((c, i) => <span key={`b${i}`} className={cellCls}>{c}</span>)}
        <span className={cellCls}>{cells.cell[0]}{hi}{cells.cell[2]}</span>
        {cells.after.map((c, i) => <span key={`a${i}`} className={cellCls}>{c}</span>)}
        {cells.hasGap && <span className="text-faint">…</span>}
      </div>
      {cells.hasGap && (
        <button onClick={() => setOpen(!open)} className="mt-2 inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted hover:text-fg-3 cursor-pointer">
          <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />{open ? 'Hide full row' : 'Show full row'}
        </button>
      )}
      {open && <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-muted">{cells.all.map((c, i) => <span key={i} className={cellCls}>{c}</span>)}</div>}
    </div>
  )
}

// One plain sentence per funnel stage, shown under the number (visible, not a tooltip).
const FUNNEL_TEXT = [
  'Papers Tavily returned for the benchmark search.',
  'Papers whose protocol and results were read from the arXiv LaTeX source (Tavily as fallback). A run reads at most three per benchmark.',
  'A result we can quote word for word from the results table.',
  'The paper splits the data the way our task does, so the numbers are comparable.',
  'A quoted claim selected for checking. A new experiment also needs a supported setup.',
]

function Funnel({ funnel, running }: { funnel: DiscoveryView['funnel']; running: boolean }) {
  const max = Math.max(1, ...funnel.map((f) => f.value))
  const reduce = useReducedMotion()
  const bar = reduce ? { duration: 0 } : { duration: 0.7, ease: 'easeOut' as const }   // bars grow, they do not jump
  return (
    <div className="space-y-3">
      {funnel.map((f, i) => (
        <div key={f.label}>
          <div className="flex items-baseline justify-between text-[12px]">
            <span className="text-fg-4">{f.label}</span>
            {running && f.value === 0
              ? <span aria-label="not computed yet" className="h-3.5 w-6 rounded bg-raised motion-safe:animate-pulse" />
              : <span className="font-mono text-fg-2">{f.value}</span>}
          </div>
          <div className="mt-0.5 text-[11px] leading-relaxed text-muted">{FUNNEL_TEXT[i]}{f.note ? ` ${f.note}.` : ''}</div>
          <div className="mt-1.5 h-1.5 rounded-sm bg-raised">
            <motion.div className={`h-full rounded-sm ${i === funnel.length - 1 ? 'bg-tone-green-solid' : 'bg-fill-lo'}`}
              initial={{ width: '0%' }} animate={{ width: `${(f.value / max) * 100}%` }} transition={bar} />
          </div>
        </div>
      ))}
    </div>
  )
}

const STATE_TEXT: Record<string, string> = {
  reading: 'Reading protocol and results',
  claim: 'Extracting claim',
  searching: 'Looking for code and community notes',
}

type CheckControl = { eligibility?: ReproductionEligibility; disabled: boolean; notice?: string; starting: boolean; error?: string; onRun: () => void }

function PaperCard({ p, active, entry, check }: { p: DPaper; active: boolean; entry?: CatalogEntry; check?: CheckControl }) {
  const t = useTween()
  const reduce = useReducedMotion()
  const c = p.claim
  const summary = useSummary(p.id)
  const rejected = p.state === 'rejected'
  const checked = p.state === 'candidate' && entry?.status === 'geprueft' && !!entry.verdict   // ran once through the loop
  const notRun = p.state === 'candidate' && (entry?.status === 'nicht_laufbar' || entry?.status === 'lauf_fehlgeschlagen' || entry?.status === 'ungeprueft' && !!entry.reason)
  const setupUnresolved = p.runnable === false || check?.eligibility?.runnable === false
  return (
    <motion.div layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: rejected ? 0.55 : 1, y: 0 }} transition={t}
      className="nx-card rounded-xl border border-line bg-surface p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-[14px] font-semibold leading-snug text-fg">{p.title}</h3>
        <span className="font-mono text-[10px] text-muted shrink-0">{p.id}</span>
      </div>

      {summary && (p.state === 'candidate' || p.state === 'searching' || p.state === 'claim') && (
        <div className="mt-3 rounded-lg border border-line/60 bg-surface/60 p-3">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
            <Sparkles className="w-3.5 h-3.5" /> AI summary
            {summary.example && <span className="rounded border border-line-strong px-1.5 py-px text-[9px] text-fg-4">Example</span>}
          </div>
          <p className="mt-1.5 text-[13px] leading-relaxed text-fg-3"><Glossed text={summary.summary} /></p>
          <p className="mt-1.5 text-[10px] leading-relaxed text-subtle">Source: {summary.source}</p>
        </div>
      )}

      {(p.state === 'reading' || p.state === 'claim' || p.state === 'searching') && (
        <div className="mt-3 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-fg-4">
          <motion.span animate={active && !reduce ? { opacity: [1, 0.4, 1] } : { opacity: 1 }}
            transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}>
            <Activity className="w-3.5 h-3.5" />
          </motion.span>
          {STATE_TEXT[p.state]}
        </div>
      )}

      {rejected && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-muted">
            <XCircle className="w-3.5 h-3.5" />
            Rejected · {stageLabel(p.stage, p.reason)}
          </div>
          <p className="mt-1.5 text-[12px] text-muted leading-relaxed">{humanReject(p.stage, p.reason)}</p>
        </div>
      )}

      {p.state === 'candidate' && (
        <div className="mt-3">
          <div className={`flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider ${setupUnresolved ? 'text-tone-amber' : 'text-tone-green'}`}>
            {setupUnresolved ? <AlertCircle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            {setupUnresolved ? 'Claim found · setup unresolved' : 'Checkable claim found'}
            <Hint text={HINTS.unverified} />
            {p.cached && (
              <span className="inline-flex items-center gap-1 font-normal normal-case tracking-normal text-subtle">from cache<Hint text={HINTS.cache} /></span>
            )}
          </div>
        </div>
      )}

      {c && (p.state === 'searching' || p.state === 'candidate') && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className={label}>Method</div>
              <div className="mt-1 text-[13px] text-fg-3">{c.method ?? '–'}</div>
            </div>
            <div>
              <div className={label}>Benchmark</div>
              <div className="mt-1 text-[13px] text-fg-3">{benchmarkText(p)}</div>
            </div>
          </div>
          <p className="m-0 flex flex-wrap items-baseline gap-3"><span className={label}>Paper claim</span><span className="font-mono text-lg font-semibold text-fg">{formatMetric(c.metric, c.target)}</span></p>
          <QuoteView quote={c.quote} target={c.target} />
          {c.column_check && (
            <p className="inline-flex items-center gap-1 text-[11px] text-muted">
              {c.column_check === 'ok' ? 'Table column verified' : 'Column not verified'}<Hint text={HINTS.column} />
            </p>
          )}
        </div>
      )}

      {p.state === 'candidate' && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line/60 pt-3">
          {p.protocol === 'passt' ? (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-tone-green"><CheckCircle2 className="w-3.5 h-3.5" />Protocol confirmed by quote<Hint text={HINTS.protocol} /></span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-tone-amber"><AlertCircle className="w-3.5 h-3.5" />Protocol not stated in text<Hint text={HINTS.protocol} /></span>
          )}
          <span className="flex flex-wrap items-center gap-4 text-[11px] text-muted">
            <span><span className="font-mono text-fg-3">{p.communityHits ?? 0}</span> community {p.communityHits === 1 ? 'note' : 'notes'}</span>
            {p.githubUrl ? (
              <a href={p.githubUrl} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}>
                <Code className={ICON_MD} />Authors’ repository<ExtArrow />
              </a>
            ) : <span className="text-subtle" title="The paper is rebuilt from its text alone">No public code – paper-only rebuild</span>}
            <a href={`https://arxiv.org/abs/${p.id}`} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}>
              <BookOpen className={ICON_MD} />arXiv<ExtArrow />
            </a>
          </span>
        </div>
      )}
      {p.state === 'candidate' && check && <div className="mt-4 border-t border-line/60 pt-3">
        {check.eligibility?.runnable === false ? <p className="m-0 text-[12px] leading-relaxed text-tone-amber">Check unavailable: {translateReason(check.eligibility.reason ?? '')}</p>
          : <>
            <button onClick={check.onRun} disabled={check.disabled || check.starting} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
              {check.starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}Check this claim
            </button>
            <p className="mb-0 mt-2 text-[11px] leading-relaxed text-muted">New experiment · 3 seeds · can take tens of minutes. Unverified setup or table columns keep the conclusion provisional.</p>
            {check.notice && <p role="status" className="mb-0 mt-2 text-[12px] text-muted">{check.notice}</p>}
          </>}
        {check.error && <p role="alert" className="mb-0 mt-2 text-[12px] text-tone-amber">{check.error}</p>}
      </div>}
      {p.state === 'candidate' && !check && <p className="mb-0 mt-3 text-[11px] text-muted">Run live discovery to save this claim for a new experiment.</p>}
      {p.state === 'candidate' && entry && (checked || notRun) && <details className="group mt-4 border-t border-line/60 pt-3" data-testid="saved-discovery-check">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-[12px] font-medium text-fg-3">
          {entry.status === 'ungeprueft' ? 'Last check stopped' : 'Saved reproduction check'}{entry.checked_at ? ` · ${new Date(entry.checked_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}` : ''}
          <ChevronDown className="h-3.5 w-3.5 shrink-0 transition-transform group-open:rotate-180" />
        </summary>
        <p className="mb-3 mt-3 text-[12px] leading-relaxed text-muted">{entry.status === 'ungeprueft'
          ? 'This attempt stopped without a paper verdict. The reason below describes what must be resolved before the claim can be assessed.'
          : 'A separate experiment saved in the catalog. This search did not run or update it. Its outcome describes our implementation; method fidelity is not verified.'}</p>
        {checked ? <div className="space-y-3"><CatalogVerdict entry={entry} /><ClaimedMeasured entry={entry} /><EvidenceLink entry={entry}>Inspect saved check</EvidenceLink></div>
          : <p className="text-[12px] text-muted">{STATUS_TEXT[entry.status]}{entry.reason ? `: ${translateReason(entry.reason)}` : ''}</p>}
      </details>}
    </motion.div>
  )
}

function Timeline({ view, runtime }: { view: DiscoveryView; runtime: number }) {
  const reduce = useReducedMotion()
  const span = Math.max(1, runtime)
  const lanes = [['tavily', 'Tavily'], ['llm', 'Model']] as const
  return (
    <div>
      <div className="space-y-2">
        {lanes.map(([kind, name]) => (
          <div key={kind} className="flex items-center gap-3">
            <span className="w-12 text-[10px] uppercase tracking-wider text-muted">{name}</span>
            <div className="relative h-4 flex-1 rounded-sm bg-raised">
              {view.timeline.filter((x) => x.kind === kind).map((x, i) => (
                <motion.div key={i} title={`${x.label}: ${(x.end - x.start).toFixed(2)}s`}
                  initial={reduce ? false : { opacity: 0, scaleX: 0.4 }} animate={{ opacity: 1, scaleX: 1 }} transition={{ duration: 0.25, ease: 'easeOut' }}
                  className={`absolute top-0 h-full origin-left rounded-sm ${kind === 'tavily' ? 'bg-fill-hi' : 'bg-fill-lo'}`}
                  style={{ left: `${((x.start - view.t0) / span) * 100}%`, width: `${Math.max(0.6, ((x.end - x.start) / span) * 100)}%` }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between pl-[60px] font-mono text-[10px] text-subtle"><span>0s</span><span>{fmtSec(span)}</span></div>
    </div>
  )
}

export function Discovery() {
  const [datasets, setDatasets] = useState<string[]>(DEFAULT_DATASETS)
  const { status: liveStatus, refresh: refreshStatus } = useLiveStatus()
  const { entries: eligibility, refresh: refreshEligibility } = useReproductionEligibility()
  const [initialReady, setInitialReady] = useState(false)
  const [starting, setStarting] = useState<string | null>(null)
  const [checkError, setCheckError] = useState<{ id: string; message: string } | null>(null)
  const interacted = useRef(false)
  const disc = statusFor(liveStatus, 'discovery')
  const paused = isPaused(disc)
  const [mode, setMode] = useState<Mode>('idle')
  const [events, setEvents] = useState<DEvent[]>([])
  const [recorded, setRecorded] = useState(false)
  const [replaySource, setReplaySource] = useState<'server' | 'example' | null>(null)
  const [offline, setOffline] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const speedRef = useRef(1)
  const stop = useRef<() => void>(() => {})
  useEffect(() => { speedRef.current = speed }, [speed])
  useEffect(() => () => stop.current(), [])
  const view = useMemo(() => deriveView(events), [events])
  const catalog = useCatalog()
  const running = mode === 'live' || mode === 'replay'
  // Real running timer: event runtime plus wall time since the last event (replay: times the speed).
  const eventRuntime = view.tEnd - view.t0
  const lastEvent = useRef({ n: -1, at: 0 })
  if (lastEvent.current.n !== events.length) lastEvent.current = { n: events.length, at: performance.now() }
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => setTick((n) => n + 1), 100)
    return () => window.clearInterval(id)
  }, [running])
  const runtime = tickedRuntime(eventRuntime, performance.now() - lastEvent.current.at, mode === 'replay' ? speed : 1, running && events.length > 0)

  const reset = () => { interacted.current = true; stop.current(); setEvents([]); setError(null); setRecorded(false); setReplaySource(null); setCheckError(null) }

  const startReplay = useCallback(async () => {
    reset()
    try {
      let all: DEvent[] = []
      let source: 'server' | 'example' = 'server'
      try {
        const response = await fetch('/api/discover/replay', { cache: 'no-store' })
        const saved = response.ok && response.headers.get('content-type')?.includes('json') ? await response.json() : []
        if (Array.isArray(saved)) all = saved.map((e) => parseLiveEvent(JSON.stringify(e))).filter((e): e is DEvent => e !== null)
      } catch { /* static mirror: use the explicitly labeled demo recording */ }
      if (!all.length) {
        source = 'example'
        const r = await fetch(import.meta.env.BASE_URL + 'data/discovery/replay.jsonl')
        const text = r.ok ? await r.text() : ''
        all = /^\s*<!doctype html/i.test(text) ? [] : parseJsonl(text)
      }
      if (!all.length) throw new Error('empty')
      setRecorded(true)
      setReplaySource(source)
      setMode('replay')
      let i = 0
      let timer: number | undefined
      let cancelled = false
      stop.current = () => { cancelled = true; window.clearTimeout(timer) }
      const step = () => {
        if (cancelled) return
        setEvents(all.slice(0, i + 1))
        i++
        if (i >= all.length) { setMode(all.some((e) => e.type === 'error') ? 'failed' : 'done'); return }
        const gap = Math.min((all[i].ts - all[i - 1].ts) * 1000, 2500) / speedRef.current
        timer = window.setTimeout(step, gap)
      }
      step()
    } catch {
      setError('The recorded run could not be loaded.')
      setMode('idle')
    }
  }, [])

  const startLive = async () => {
    reset()
    setMode('live')
    let res: Response
    try {
      res = await fetch('/api/discover', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ datasets }),
      })
    } catch {
      setError('The live backend is not reachable.')
      setMode('idle')
      return
    }
    if (!res.ok) {
      let body: unknown = null
      try { body = await res.json() } catch { /* no body */ }
      refreshStatus()
      setError(res.status === 404 ? 'The live backend is not available on this site.'
        : res.status === 503 || res.status === 429 ? startErrorText(res.status, body, 'search').text
        : `The live backend answered with an error (${res.status}).`)
      setMode('idle')
      return
    }
    const { job_id } = await res.json()
    // A proxy or network drop closes the stream while the search keeps running: reconnect from the last received event.
    let failed = false
    let closed = false
    let received = 0
    let tries = 0
    let es: EventSource | null = null
    let timer: number | undefined
    stop.current = () => { closed = true; es?.close(); window.clearTimeout(timer) }
    const connect = () => {
      if (closed) return
      es = new EventSource(`/api/discover/${encodeURIComponent(job_id)}/stream?from=${received}`)
      es.onmessage = (m) => {
        tries = 0
        const e = parseLiveEvent(m.data)
        received++
        if (!e) return
        if (e.type === 'done') {
          closed = true
          es?.close()
          setEvents((x) => [...x, e])
          setMode(failed ? 'failed' : 'done')
          refreshStatus()
          refreshEligibility()
          return
        }
        if (e.type === 'error') { failed = true; setError(String(e.message || 'The discovery failed.')) }
        setEvents((x) => [...x, e])
      }
      es.onerror = () => {
        es?.close()
        if (closed) return
        if (++tries <= 6) { timer = window.setTimeout(connect, 1500); return }
        closed = true
        setError((previous) => previous ?? 'The live stream ended before completion was confirmed. The search may still be running; check its status before retrying.')
        setMode('failed')
        refreshStatus()
      }
    }
    connect()
  }

  // No API here (static mirror) or live search paused: open on the last stored run instead of an empty page.
  useEffect(() => {
    let alive = true
    fetchJobInfo().then(async (info) => {
      if (!alive || interacted.current) return
      if (info === null) { setOffline(true); startReplay(); return }
      try {
        const response = await fetch('/api/discover/candidates', { cache: 'no-store' })
        const saved = response.ok && response.headers.get('content-type')?.includes('json') ? await response.json() : []
        if (alive && !interacted.current && Array.isArray(saved) && saved.length) {
          setEvents(saved.map((e) => parseLiveEvent(JSON.stringify(e))).filter((e): e is DEvent => e !== null))
          setMode('saved')
        }
      } catch { /* a fresh install has no saved claims */ }
      finally { if (alive) setInitialReady(true) }
    })
    return () => { alive = false }
  }, [startReplay])
  useEffect(() => { if (initialReady && paused && mode === 'idle' && !events.length) startReplay() }, [initialReady, paused, mode, events.length, startReplay])

  const runClaim = async (id: string) => {
    setStarting(id); setCheckError(null)
    try {
      const jobId = await startReproduction(id)
      window.location.assign(`#/live/${jobId}`)
    } catch (err) {
      setCheckError({ id, message: err instanceof Error ? err.message : 'Could not start the check.' })
      refreshStatus(); refreshEligibility()
    } finally { setStarting(null) }
  }
  const repro = statusFor(liveStatus, 'reproduction')
  const checkNotice = offline ? 'Live checks are not available on this site.'
    : running || repro?.reason === 'busy' ? 'Another run is active. Wait for it to finish.'
    : isPaused(repro) ? `Live checks are paused (${repro?.reason === 'daily_limit' ? 'daily limit' : 'budget limit'}). ${resumeText(repro?.reason, repro?.reset_at) ?? ''}` : undefined

  const toggle = (d: string) => setDatasets((x) => (x.includes(d) ? x.filter((y) => y !== d) : [...x, d]))
  const status = mode === 'saved' ? 'Saved claims' : running ? 'Running' : mode === 'failed' ? (view.tavilyCalls ? 'Finished with errors' : 'Failed') : mode === 'done' ? 'Finished' : 'Ready'
  const displayedError = error ?? view.notes.find((note) => note.kind === 'error')?.message
  const papers = [...view.papers].reverse()
  const candidates = papers.filter((p) => p.state === 'candidate')
  const inProgress = papers.filter((p) => p.state !== 'candidate' && p.state !== 'rejected')
  const rejected = papers.filter((p) => p.state === 'rejected')
  const example = catalog?.filter((entry) => entry.run_id && entry.jury?.ready && entry.verdict?.urteil === 'reproduziert')
    .sort((a, b) => (b.checked_at ?? '').localeCompare(a.checked_at ?? ''))[0]

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <PageIntro kicker="Discover" title="Find papers worth checking"
        compact="Choose supported datasets, then watch a recorded search or start a live one."
        what="Tavily finds papers with quoted claims on supported datasets. Discovery checks their sources and setup to find candidates for an experiment. Reproduction is a separate step."
        look="Watch a recorded search or start a live one. Choose Check this claim to run a new experiment for a saved candidate and inspect its evidence." />

      <div className="nx-card mb-6 rounded-xl border border-line bg-surface p-6">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
          <div>
            <div className={label}>Benchmarks</div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
              {DATASETS.map((d) => (
                <label key={d} className={`${INLINE} select-none rounded-lg border px-3 py-1.5 text-sm transition-colors ${running ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} ${datasets.includes(d) ? 'border-line-stronger bg-raised text-fg' : 'border-line text-fg-4 hover:border-line-strong hover:text-fg-3'} focus-within:outline focus-within:outline-1 focus-within:outline-zinc-500`}>
                  <input type="checkbox" checked={datasets.includes(d)} onChange={() => toggle(d)} disabled={running} className="sr-only" />
                  <Check className={`h-3.5 w-3.5 shrink-0 ${datasets.includes(d) ? 'opacity-100' : 'opacity-0'}`} aria-hidden />
                  {d}
                </label>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <button onClick={startLive} disabled={running || !datasets.length || paused || offline || disc?.reason === 'busy'} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
              <Play className="w-4 h-4" />Start live run
            </button>
            <button onClick={startReplay} disabled={running} className={`${btn} border-line text-fg-3 hover:bg-raised-2`}>
              <RotateCcw className="w-4 h-4" />Replay recorded run
            </button>
          </div>
          <div>
            <div className={label}>Replay speed</div>
            <div className="mt-2 flex gap-1">
              {[1, 4].map((s) => (
                <button key={s} onClick={() => setSpeed(s)}
                  className={`px-3 py-1.5 font-mono text-xs rounded-md border cursor-pointer ${speed === s ? 'border-line-stronger text-fg' : 'border-line text-muted hover:text-fg-3'}`}>{s}x</button>
              ))}
            </div>
          </div>
        </div>

        <p className="mb-0 mt-4 text-pretty text-[12px] leading-relaxed text-muted">
          Search covers the selected datasets; this demo has no paper-URL input. A live run spends search and model calls, with one run at a time and a daily limit. Replay has no limit.
        </p>
        {paused && (
          <p role="status" className="mb-0 mt-3 flex max-w-3xl items-start gap-2 text-[13px] leading-relaxed text-fg-3">
            <PauseCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
            <span>Live search is paused ({disc?.reason === 'daily_limit' ? 'daily limit' : 'budget limit'}). {resumeText(disc?.reason, disc?.reset_at)}</span>
          </p>
        )}
        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line/60 pt-4 text-[12px] text-muted">
          <span role="status" className={`${INLINE} font-medium uppercase tracking-wider ${running ? 'text-tone-green' : 'text-fg-3'}`}>
            {running
              ? <span className="relative flex h-2 w-2" aria-hidden><span className="absolute inline-flex h-full w-full rounded-full bg-tone-green-solid opacity-60 motion-safe:animate-ping" /><span className="relative inline-flex h-2 w-2 rounded-full bg-tone-green-solid" /></span>
              : mode === 'failed' ? <XCircle className={`${ICON_MD} text-tone-red`} aria-hidden /> : mode === 'done' ? <CheckCircle2 className={ICON_MD} aria-hidden /> : <span className="h-2 w-2 rounded-full bg-line-stronger" aria-hidden />}
            {status}{running && mode === 'replay' && <span className="font-normal normal-case tracking-normal text-muted">· replay</span>}
          </span>
          {mode !== 'saved' && <span>Runtime <span className="font-mono text-fg-3">{fmtSec(runtime)}</span></span>}
          <span>Papers <span className="font-mono text-fg-3">{view.papers.length}</span></span>
          <span>Candidates <span className="font-mono text-fg-3">{view.funnel[view.funnel.length - 1].value}</span></span>
          {recorded && events.length > 0 && <span>{replaySource === 'server' ? 'Saved server search' : 'Demo recording'} from {fmtDate(events[0].ts)}</span>}
        </div>

        {recorded && (offline || paused) && (
          <p role="status" className="mb-0 mt-4 flex items-start gap-2 text-[13px] leading-relaxed text-fg-3">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
            <span>{offline ? 'Live search is not available on this site.' : 'Live search is paused.'} Showing {replaySource === 'server' ? 'the latest saved server search' : 'the bundled demo recording'}{events.length ? `, recorded ${fmtDate(events[0].ts)}` : ''}.</span>
          </p>
        )}
        {displayedError && (
          <div role="alert" className="mt-4 flex flex-wrap items-center gap-4 text-[13px] text-tone-amber">
            <span className="inline-flex items-center gap-2"><AlertCircle className="w-4 h-4" />{displayedError}</span>
            <button onClick={startReplay} className="text-fg-2 underline underline-offset-4 cursor-pointer hover:text-fg-max">Show recorded run instead</button>
          </div>
        )}
        {view.notes.filter((note) => note.kind === 'warning').map((note, i) => <p key={i} className="mb-0 mt-3 text-[12px] text-tone-amber">{note.message}</p>)}
      </div>

      {!events.length && mode === 'idle' ? (
        <div className="rounded-xl border border-line bg-surface p-8 text-center">
          <p className="font-grotesk m-0 text-2xl font-semibold tracking-[-0.03em] text-fg">Nothing has run yet</p>
          <p className="mx-auto mb-0 mt-3 max-w-xl text-[15px] leading-relaxed text-fg-4">
            Start with the recorded run: on the left you will see every Tavily call as it happens, on the right the funnel from found papers down to checkable claims.
          </p>
          <button onClick={startReplay} className={`${btn} mt-6 border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}><RotateCcw className="w-4 h-4" />Replay recorded run</button>
        </div>
      ) : (
      <div className="grid gap-6 lg:grid-cols-2 items-start">
        {mode === 'saved' ? <Section title="Saved live discovery" hint="no new API calls">
          <p className="text-[13px] leading-relaxed text-muted">These claims were found by earlier live searches and saved on this server. Choose a compatible claim to start a new experiment. Reading this list does not start a search or a check.</p>
        </Section> : <Section title="Tavily research feed" hint={recorded ? 'recorded search · no new API calls' : 'live API calls'}>
          <div className="mb-4 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-muted">
            <span>Tavily calls <span className="font-mono text-fg-3">{view.tavilyCalls}</span></span>
            <span>Model calls <span className="font-mono text-fg-3">{view.llmCalls}</span>
              {view.cachedLlmCalls > 0 && <span className="text-subtle"> ({view.cachedLlmCalls} cached, no cost)</span>}</span>
            <span>Model cost <span className="font-mono text-fg-3">${view.costUsd.toFixed(3)}</span></span>
          </div>
          <div className="space-y-3">
            {running && <div aria-hidden className="h-[104px] rounded-xl border border-dashed border-line bg-panel/30 p-4 motion-safe:animate-pulse"><div className="h-2.5 w-24 rounded bg-raised" /><div className="mt-4 h-2.5 w-3/4 rounded bg-raised" /></div>}
            <AnimatePresence initial={false}>
              {[...view.feed].reverse().map((c) => <FeedCard key={c.seq} call={c} recorded={recorded} />)}
            </AnimatePresence>
            {!view.feed.length && !running && <p className="text-[13px] text-subtle">No Tavily calls yet. Start a run or replay the recorded one.</p>}
          </div>
        </Section>}

        <div className="space-y-6">
          {mode !== 'saved' && <Section title="Pipeline">
            <Funnel funnel={view.funnel} running={running} />
          </Section>}
          <Section title="Claims to check" hint={`${candidates.length} found · ${papers.length} papers seen`}>
            <p className="mb-4 text-[12px] leading-relaxed text-muted">These claims passed the discovery filters. Discovery reads the claims; Check this claim starts a separate experiment. Saved reproduction checks are available below each matching claim.</p>
            <div className="space-y-3">
              {[...candidates, ...inProgress].map((p) => {
                const entry = savedCheckForClaim(p, catalog ?? [])
                const id = p.catalogId ?? entry?.id
                const eligible = id ? eligibility?.[id] ?? (p.runnable !== undefined ? { runnable: p.runnable, reason: p.runReason ?? null, warnings: [] } : undefined) : undefined
                return <PaperCard key={p.catalogId ?? p.id} p={p} active={running && view.activeId === p.id} entry={entry}
                  check={id && (p.catalogId || eligible) ? { eligibility: eligible, disabled: !!checkNotice || starting !== null, notice: checkNotice,
                    starting: starting === id, error: checkError?.id === id ? checkError.message : undefined, onRun: () => runClaim(id) } : undefined} />
              })}
              {rejected.length > 0 && <details className="mt-4 border-t border-line pt-3">
                <summary className="cursor-pointer text-[12px] text-muted">Papers outside the discovery criteria ({rejected.length})</summary>
                <p className="text-[12px] leading-relaxed text-muted">No compatible claim was selected from these papers. This is a search-filter result, not an experiment verdict.</p>
                <div className="mt-3 space-y-3">{rejected.map((p) => <PaperCard key={p.id} p={p} active={false} />)}</div>
              </details>}
              {!papers.length && running && [0, 1].map((i) => (
                <div key={i} aria-hidden className="rounded-xl border border-dashed border-line p-5 motion-safe:animate-pulse"><div className="h-3 w-2/3 rounded bg-raised" /><div className="mt-3 h-2.5 w-1/3 rounded bg-raised" /></div>
              ))}
              {!candidates.length && !inProgress.length && !running && <p className="text-[13px] text-subtle">No compatible claim was selected in this search. The filter details above explain why.</p>}
            </div>
          </Section>
          {mode !== 'saved' && <Section title="Call timeline" hint="Tavily vs. model">
            {view.timeline.length || running ? <Timeline view={view} runtime={runtime} /> : <p className="text-[13px] text-subtle">The timeline fills during the run.</p>}
          </Section>}
        </div>
      </div>
      )}
      <NextStep title="See the experiment stage"
        text={example ? `${example.claim.method ?? 'This paper'} is a separate recorded catalog example whose target was reached. Inspect its sources, generated experiment and measurements. Method fidelity is not verified.` : 'Open the catalog for saved experiments, their measurements and all outcomes.'}
        label={example ? 'Open recorded example' : 'Open the catalog'} href={example ? evidenceHref(example)! : '#catalog'} />
    </main>
  )
}

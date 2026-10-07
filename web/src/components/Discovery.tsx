import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Activity, CheckCircle2, CircleDashed, ChevronDown, ExternalLink, FileText, Play, Radio, RotateCcw, Search, XCircle, AlertCircle, BookOpen, Code, Sparkles, PauseCircle } from 'lucide-react'
import { Section } from './ui'
import { Hint } from './Hint'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, useCatalog, type CatalogEntry } from '../catalog'
import { translateReason } from '../labels'
import { useLiveStatus } from '../live'
import { isPaused, pausedText, resumeText, startErrorText, statusFor } from '../livestatus'
import {
  deriveView, domainOf, formatMetric, humanReject, parseJsonl, quoteCells, quoteExcerpt, shorten, stageLabel, benchmarkText,
  type DEvent, type DPaper, type DiscoveryView, type FeedCall,
} from '../discovery'
import { Glossed } from './Glossed'
import { NextStep, PageIntro } from './page'
import { useSummary } from '../summaries'

type Mode = 'idle' | 'live' | 'replay' | 'done'

const DATASETS = ['Cora', 'ETTh1']
const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'
const btn = 'inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed'

const PURPOSE_TITLE: Record<string, string> = { find_papers: 'Find papers', read_paper: 'Read paper', find_code: 'Find code', community: 'Community' }

const fmtDate = (ts: number) => new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
const fmtSec = (s: number) => `${Math.max(0, s).toFixed(1)}s`

const HINTS = {
  protocol: 'Protocol = how the paper split the data (and, for forecasting, input length and horizon). Numbers are only comparable when our re-run uses the same split.',
  column: "We check that the number sits in the proposed method's column of the results table, not in a baseline column.",
  unverified: 'Nullius has only read this claim. It has not re-run the experiment yet.',
  cache: 'Read in an earlier run; no new API calls.',
}

function useTween() {
  const reduce = useReducedMotion()
  return reduce ? { duration: 0 } : { duration: 0.2, ease: 'easeOut' as const }
}

function FeedCard({ call }: { call: FeedCall }) {
  const [open, setOpen] = useState(false)
  const t = useTween()
  const Icon = call.endpoint === 'search' ? Search : FileText
  return (
    <motion.div layout="position" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={t}
      className="rounded-xl border border-line bg-panel/50 hover:border-line-strong transition-colors">
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
        <p className="mt-2 text-[10px] uppercase tracking-wider text-subtle">Live Tavily API call</p>
      </button>
      {open && call.topHits.length > 0 && (
        <ul className="border-t border-line/60 px-4 py-3 space-y-2">
          {call.topHits.map((h) => (
            <li key={h.url} className="text-[12px] leading-snug">
              <a href={h.url} target="_blank" rel="noreferrer" className="text-fg-3 hover:text-fg-max inline-flex items-start gap-1.5">
                <ExternalLink className="w-3 h-3 mt-0.5 shrink-0 text-muted" />
                <span>{h.title || h.url}</span>
              </a>
              <div className="font-mono text-[10px] text-subtle ml-4">{domainOf(h.url)}</div>
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
  'Papers whose protocol and results Tavily read in full. A run reads at most three per benchmark.',
  'A result we can quote word for word from the results table.',
  'The paper splits the data the way our task does, so the numbers are comparable.',
  'A claim we can re-run. The catalog lists the ones that have been run.',
]

function Funnel({ funnel }: { funnel: DiscoveryView['funnel'] }) {
  const max = Math.max(1, ...funnel.map((f) => f.value))
  const t = useTween()
  return (
    <div className="space-y-3">
      {funnel.map((f, i) => (
        <div key={f.label}>
          <div className="flex items-baseline justify-between text-[12px]">
            <span className="text-fg-4">{f.label}</span>
            <span className="font-mono text-fg-2">{f.value}</span>
          </div>
          <div className="mt-0.5 text-[11px] leading-relaxed text-muted">{FUNNEL_TEXT[i]}{f.note ? ` ${f.note}.` : ''}</div>
          <div className="mt-1.5 h-1.5 rounded-sm bg-raised">
            <motion.div className={`h-full rounded-sm ${i === funnel.length - 1 ? 'bg-fill-max' : 'bg-fill-lo'}`}
              initial={false} animate={{ width: `${(f.value / max) * 100}%` }} transition={t} />
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

function PaperCard({ p, active, entry }: { p: DPaper; active: boolean; entry?: CatalogEntry }) {
  const t = useTween()
  const reduce = useReducedMotion()
  const c = p.claim
  const summary = useSummary(p.id)
  const rejected = p.state === 'rejected'
  const checked = p.state === 'candidate' && entry?.status === 'geprueft' && !!entry.verdict   // ran once through the loop
  const notRun = p.state === 'candidate' && (entry?.status === 'nicht_laufbar' || entry?.status === 'lauf_fehlgeschlagen')
  return (
    <motion.div layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: rejected ? 0.55 : 1, y: 0 }} transition={t}
      className="rounded-xl border border-line bg-surface p-5">
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

      {checked && entry && (
        <div className="mt-3">
          <CatalogVerdict entry={entry} />
        </div>
      )}

      {p.state === 'candidate' && !checked && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-fg-3">
            <CircleDashed className="w-3.5 h-3.5" />
            Unverified candidate
            <Hint text={HINTS.unverified} />
            {p.cached && (
              <span className="inline-flex items-center gap-1 font-normal normal-case tracking-normal text-subtle">from cache<Hint text={HINTS.cache} /></span>
            )}
          </div>
          {notRun && entry && (
            <p className="mt-1.5 text-[11px] leading-relaxed text-muted">{STATUS_TEXT[entry.status]}{entry.reason ? `: ${translateReason(entry.reason)}` : ''}</p>
          )}
        </div>
      )}

      {c && (p.state === 'searching' || p.state === 'candidate') && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className={label}>{checked ? 'Method' : (c.method ?? 'Method')}</div>
              {checked   // the claimed number is shown once, in CLAIMED below
                ? <div className="mt-1 text-[13px] text-fg-3">{c.method ?? '–'}</div>
                : <div className="mt-1 font-mono text-xl font-semibold text-fg-strong">{formatMetric(c.metric, c.target)}</div>}
            </div>
            <div>
              <div className={label}>Benchmark</div>
              <div className="mt-1 text-[13px] text-fg-3">{benchmarkText(p)}</div>
            </div>
          </div>
          <QuoteView quote={c.quote} target={c.target} />
          {checked && entry && <ClaimedMeasured entry={entry} />}
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
            {checked && entry && <EvidenceLink entry={entry}>View evidence</EvidenceLink>}
            <span><span className="font-mono text-fg-3">{p.communityHits ?? 0}</span> community {p.communityHits === 1 ? 'note' : 'notes'}</span>
            {p.githubUrl ? (
              <a href={p.githubUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-3 hover:text-fg-max">
                <Code className="w-3.5 h-3.5" />Code
              </a>
            ) : <span className="text-subtle">No code found</span>}
            <a href={`https://arxiv.org/abs/${p.id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-3 hover:text-fg-max">
              <BookOpen className="w-3.5 h-3.5" />arXiv
            </a>
          </span>
        </div>
      )}
    </motion.div>
  )
}

function Timeline({ view }: { view: DiscoveryView }) {
  const span = Math.max(1, view.tEnd - view.t0)
  const lanes = [['tavily', 'Tavily'], ['llm', 'Model']] as const
  return (
    <div>
      <div className="space-y-2">
        {lanes.map(([kind, name]) => (
          <div key={kind} className="flex items-center gap-3">
            <span className="w-12 text-[10px] uppercase tracking-wider text-muted">{name}</span>
            <div className="relative h-4 flex-1 rounded-sm bg-raised">
              {view.timeline.filter((x) => x.kind === kind).map((x, i) => (
                <div key={i} title={`${x.label}: ${(x.end - x.start).toFixed(2)}s`}
                  className={`absolute top-0 h-full rounded-sm ${kind === 'tavily' ? 'bg-fill-hi' : 'bg-fill-lo'}`}
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
  const [datasets, setDatasets] = useState<string[]>(DATASETS)
  const { status: liveStatus, refresh: refreshStatus } = useLiveStatus()
  const disc = statusFor(liveStatus, 'discovery')
  const paused = isPaused(disc)
  const [mode, setMode] = useState<Mode>('idle')
  const [events, setEvents] = useState<DEvent[]>([])
  const [recorded, setRecorded] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const speedRef = useRef(1)
  const stop = useRef<() => void>(() => {})
  useEffect(() => { speedRef.current = speed }, [speed])
  useEffect(() => () => stop.current(), [])
  const view = useMemo(() => deriveView(events), [events])
  const catalog = useCatalog()
  const running = mode === 'live' || mode === 'replay'

  const reset = () => { stop.current(); setEvents([]); setError(null); setRecorded(false) }

  const startReplay = useCallback(async () => {
    reset()
    try {
      const r = await fetch(import.meta.env.BASE_URL + 'data/discovery/replay.jsonl')
      const text = r.ok ? await r.text() : ''
      const all = /^\s*<!doctype html/i.test(text) ? [] : parseJsonl(text)
      if (!all.length) throw new Error('empty')
      setRecorded(true)
      setMode('replay')
      let i = 0
      let timer: number | undefined
      let cancelled = false
      stop.current = () => { cancelled = true; window.clearTimeout(timer) }
      const step = () => {
        if (cancelled) return
        setEvents(all.slice(0, i + 1))
        i++
        if (i >= all.length) { setMode('done'); return }
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
    const es = new EventSource(`/api/discover/${job_id}/stream`)
    let received = 0
    stop.current = () => es.close()
    es.onmessage = (m) => {
      try { const e = JSON.parse(m.data) as DEvent; received++; setEvents((x) => [...x, e]) } catch { /* ignore keepalive noise */ }
    }
    es.onerror = () => {
      es.close()
      if (received === 0) setError('The live stream could not be opened.')
      setMode(received === 0 ? 'idle' : 'done')
    }
  }

  const toggle = (d: string) => setDatasets((x) => (x.includes(d) ? x.filter((y) => y !== d) : [...x, d]))
  const status = mode === 'live' ? 'Live' : mode === 'replay' ? 'Replay' : mode === 'done' ? 'Finished' : 'Ready'
  const papers = [...view.papers].reverse()

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <PageIntro kicker="Discover" title="Find papers worth checking"
        what="Tavily searches the web for new papers that report a result on a benchmark we can rebuild. It reads only the protocol and results, and keeps claims it can quote word for word."
        look="Press Replay recorded run to watch a real search play back, Papers appear on the right as they are found, each with a short summary." />

      <div className="mb-6 rounded-xl border border-line bg-surface p-6">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
          <div>
            <div className={label}>Benchmarks</div>
            <div className="mt-2 flex gap-4">
              {DATASETS.map((d) => (
                <label key={d} className="inline-flex items-center gap-2 text-sm text-fg-3 cursor-pointer">
                  <input type="checkbox" checked={datasets.includes(d)} onChange={() => toggle(d)} disabled={running} className="accent-fg-3" />
                  {d}
                </label>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <button onClick={startLive} disabled={running || !datasets.length || paused} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
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

        <p className="mb-0 mt-4 max-w-3xl text-[12px] leading-relaxed text-muted">
          A live run searches the web right now and spends search and model calls, so it is limited: one run at a time, and a daily limit. Replay has no limit.
        </p>
        {paused && (
          <p role="status" className="mb-0 mt-3 flex max-w-3xl items-start gap-2 text-[13px] leading-relaxed text-fg-3">
            <PauseCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
            <span>{pausedText(disc?.reason, 'search')} {resumeText(disc?.reason, disc?.reset_at)}</span>
          </p>
        )}
        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-line/60 pt-4 text-[12px] text-muted">
          <span className={`inline-flex items-center gap-1.5 font-medium uppercase tracking-wider ${mode === 'live' ? 'text-tone-green' : 'text-fg-3'}`}>
            <Radio className="w-3.5 h-3.5" />{status}
          </span>
          <span>Runtime <span className="font-mono text-fg-3">{fmtSec(view.tEnd - view.t0)}</span></span>
          <span>Papers <span className="font-mono text-fg-3">{view.papers.length}</span></span>
          <span>Candidates <span className="font-mono text-fg-3">{view.funnel[view.funnel.length - 1].value}</span></span>
          {recorded && events.length > 0 && <span>Recorded run from {fmtDate(events[0].ts)}</span>}
        </div>

        {error && (
          <div className="mt-4 flex flex-wrap items-center gap-4 text-[13px] text-tone-amber">
            <span className="inline-flex items-center gap-2"><AlertCircle className="w-4 h-4" />{error}</span>
            <button onClick={startReplay} className="text-fg-2 underline underline-offset-4 cursor-pointer hover:text-fg-max">Show recorded run instead</button>
          </div>
        )}
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
        <Section title="Tavily research feed" hint="live API calls">
          <div className="mb-4 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-muted">
            <span>Tavily calls <span className="font-mono text-fg-3">{view.tavilyCalls}</span></span>
            <span>Model calls <span className="font-mono text-fg-3">{view.llmCalls}</span>
              {view.cachedLlmCalls > 0 && <span className="text-subtle"> ({view.cachedLlmCalls} cached, no cost)</span>}</span>
            <span>Model cost <span className="font-mono text-fg-3">${view.costUsd.toFixed(3)}</span></span>
          </div>
          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {[...view.feed].reverse().map((c) => <FeedCard key={c.seq} call={c} />)}
            </AnimatePresence>
            {!view.feed.length && <p className="text-[13px] text-subtle">No Tavily calls yet. Start a run or replay the recorded one.</p>}
          </div>
        </Section>

        <div className="space-y-6">
          <Section title="Pipeline">
            <Funnel funnel={view.funnel} />
          </Section>
          <Section title="Papers" hint={`${papers.length} seen`}>
            <p className="mb-4 text-[12px] leading-relaxed text-muted">Candidates are claims we can re-run. Unless a result is shown, a candidate has not been re-run yet.</p>
            <div className="space-y-3">
              {papers.map((p) => <PaperCard key={p.id} p={p} active={running && view.activeId === p.id} entry={catalog?.find((e) => e.arxiv_id === p.id)} />)}
              {!papers.length && <p className="text-[13px] text-subtle">Papers appear here as Tavily finds and reads them.</p>}
            </div>
          </Section>
          <Section title="Call timeline" hint="Tavily vs. model">
            {view.timeline.length ? <Timeline view={view} /> : <p className="text-[13px] text-subtle">The timeline fills during the run.</p>}
          </Section>
        </div>
      </div>
      )}
      <NextStep title="See what the checks found" text="Candidates that were run through the full loop are listed in the catalog, each with the claimed number, the measured number and its evidence."
        label="Open the catalog" href="#catalog" />
    </main>
  )
}

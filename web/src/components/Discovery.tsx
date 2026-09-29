import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Activity, CheckCircle2, CircleDashed, ChevronDown, ExternalLink, FileText, Play, Radio, RotateCcw, Search, XCircle, AlertCircle, BookOpen, Code, Info } from 'lucide-react'
import { Section } from './ui'
import {
  deriveView, domainOf, formatMetric, humanReject, parseJsonl, quoteCells, quoteExcerpt, shorten, stageLabel, benchmarkText,
  type DEvent, type DPaper, type DiscoveryView, type FeedCall,
} from '../discovery'

type Mode = 'idle' | 'live' | 'replay' | 'done'

const DATASETS = ['Cora', 'ETTh1']
const label = 'text-[10px] font-semibold uppercase tracking-widest text-zinc-500'
const btn = 'inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed'

const PURPOSE_TITLE: Record<string, string> = { find_papers: 'Find papers', read_paper: 'Read paper', find_code: 'Find code', community: 'Community' }

const fmtDate = (ts: number) => new Date(ts * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
const fmtSec = (s: number) => `${Math.max(0, s).toFixed(1)}s`

// Info icon with a short explanation: hover (mouse), tap (touch) and keyboard focus; Escape or a second tap closes it.
function Hint({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ left: 0, width: 240 })
  const ref = useRef<HTMLButtonElement>(null)
  // Keep the popover inside the viewport (16 px margin), also on narrow phones.
  const show = (on: boolean) => {
    if (on && ref.current) {
      const r = ref.current.getBoundingClientRect()
      const width = Math.min(240, window.innerWidth - 32)
      setPos({ width, left: Math.min(Math.max(r.left, 16), window.innerWidth - 16 - width) - r.left })
    }
    setOpen(on)
  }
  return (
    <span className="relative inline-flex align-middle normal-case tracking-normal font-normal">
      <button ref={ref} type="button" aria-label={text} aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); show(!open) }}
        onPointerEnter={(e) => { if (e.pointerType === 'mouse') show(true) }}
        onPointerLeave={(e) => { if (e.pointerType === 'mouse') show(false) }}
        onFocus={(e) => { if (e.currentTarget.matches(':focus-visible')) show(true) }}
        onBlur={() => show(false)}
        onKeyDown={(e) => { if (e.key === 'Escape') show(false) }}
        className="p-0.5 text-zinc-500 hover:text-zinc-300 focus-visible:text-zinc-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-zinc-500 rounded cursor-pointer">
        <Info className="w-3.5 h-3.5" />
      </button>
      {open && (
        <span role="tooltip" style={{ left: pos.left, width: pos.width }}
          className="absolute top-full z-20 mt-1.5 rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-[11px] leading-relaxed text-zinc-300 shadow-lg">
          {text}
        </span>
      )}
    </span>
  )
}

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
      className="rounded-xl border border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 transition-colors">
      <button onClick={() => setOpen(!open)} className="w-full text-left p-4 cursor-pointer">
        <div className="flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-zinc-300">
            <Icon className="w-3.5 h-3.5" />
            {PURPOSE_TITLE[call.purpose ?? ''] ?? call.endpoint}
            {call.purpose && <span className="text-zinc-600">{call.endpoint}</span>}
          </span>
          <span className="flex items-center gap-3 font-mono text-[11px] text-zinc-500">
            <span>{call.hits} {call.hits === 1 ? 'hit' : 'hits'}</span>
            <span>{(call.durationMs / 1000).toFixed(2)}s</span>
            <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
          </span>
        </div>
        <p className="mt-2 font-mono text-[12px] leading-relaxed text-zinc-400 break-all">
          {call.purpose === 'find_code' || call.purpose === 'community' ? <span className="text-zinc-600">Paper: </span> : null}
          {shorten(call.query, 110)}
        </p>
        <p className="mt-2 text-[10px] uppercase tracking-wider text-zinc-600">Live Tavily API call</p>
      </button>
      {open && call.topHits.length > 0 && (
        <ul className="border-t border-zinc-800/60 px-4 py-3 space-y-2">
          {call.topHits.map((h) => (
            <li key={h.url} className="text-[12px] leading-snug">
              <a href={h.url} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-white inline-flex items-start gap-1.5">
                <ExternalLink className="w-3 h-3 mt-0.5 shrink-0 text-zinc-500" />
                <span>{h.title || h.url}</span>
              </a>
              <div className="font-mono text-[10px] text-zinc-600 ml-4">{domainOf(h.url)}</div>
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
      <blockquote className="border-l-2 border-zinc-800 pl-3 font-mono text-[11px] leading-relaxed text-zinc-500 break-words">
        {b}<span className="text-zinc-50 font-semibold underline decoration-zinc-500 underline-offset-2">{n}</span>{a}
      </blockquote>
    )
  }
  const hi = <span className="text-zinc-50 font-semibold underline decoration-zinc-500 underline-offset-2">{cells.cell[1]}</span>
  return (
    <div className="border-l-2 border-zinc-800 pl-3 font-mono text-[11px] leading-relaxed text-zinc-500">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {cells.head.map((c, i) => <span key={`h${i}`} className={`${cellCls} text-zinc-400`}>{c}</span>)}
        {cells.hasGap && <span className="text-zinc-700">…</span>}
        {cells.before.map((c, i) => <span key={`b${i}`} className={cellCls}>{c}</span>)}
        <span className={cellCls}>{cells.cell[0]}{hi}{cells.cell[2]}</span>
        {cells.after.map((c, i) => <span key={`a${i}`} className={cellCls}>{c}</span>)}
        {cells.hasGap && <span className="text-zinc-700">…</span>}
      </div>
      {cells.hasGap && (
        <button onClick={() => setOpen(!open)} className="mt-2 inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500 hover:text-zinc-300 cursor-pointer">
          <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />{open ? 'Hide full row' : 'Show full row'}
        </button>
      )}
      {open && <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-zinc-500">{cells.all.map((c, i) => <span key={i} className={cellCls}>{c}</span>)}</div>}
    </div>
  )
}

function Funnel({ funnel }: { funnel: DiscoveryView['funnel'] }) {
  const max = Math.max(1, ...funnel.map((f) => f.value))
  const t = useTween()
  return (
    <div className="space-y-3">
      {funnel.map((f, i) => (
        <div key={f.label}>
          <div className="flex items-baseline justify-between text-[12px]">
            <span className="text-zinc-400">{f.label}</span>
            <span className="font-mono text-zinc-200">{f.value}</span>
          </div>
          {f.note && <div className="mt-0.5 text-[11px] text-zinc-600">{f.note}</div>}
          <div className="mt-1.5 h-1.5 rounded-sm bg-zinc-900">
            <motion.div className={`h-full rounded-sm ${i === funnel.length - 1 ? 'bg-zinc-200' : 'bg-zinc-600'}`}
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

function PaperCard({ p, active }: { p: DPaper; active: boolean }) {
  const t = useTween()
  const reduce = useReducedMotion()
  const c = p.claim
  const rejected = p.state === 'rejected'
  return (
    <motion.div layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: rejected ? 0.55 : 1, y: 0 }} transition={t}
      className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-[14px] font-semibold leading-snug text-zinc-100">{p.title}</h3>
        <span className="font-mono text-[10px] text-zinc-500 shrink-0">{p.id}</span>
      </div>

      {(p.state === 'reading' || p.state === 'claim' || p.state === 'searching') && (
        <div className="mt-3 flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-zinc-400">
          <motion.span animate={active && !reduce ? { opacity: [1, 0.4, 1] } : { opacity: 1 }}
            transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}>
            <Activity className="w-3.5 h-3.5" />
          </motion.span>
          {STATE_TEXT[p.state]}
        </div>
      )}

      {rejected && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            <XCircle className="w-3.5 h-3.5" />
            Rejected · {stageLabel(p.stage, p.reason)}
          </div>
          <p className="mt-1.5 text-[12px] text-zinc-500 leading-relaxed">{humanReject(p.stage, p.reason)}</p>
        </div>
      )}

      {p.state === 'candidate' && (
        <div className="mt-3">
          <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-zinc-300">
            <CircleDashed className="w-3.5 h-3.5" />
            Unverified candidate
            <Hint text={HINTS.unverified} />
            {p.cached && (
              <span className="inline-flex items-center gap-1 font-normal normal-case tracking-normal text-zinc-600">from cache<Hint text={HINTS.cache} /></span>
            )}
          </div>
        </div>
      )}

      {c && (p.state === 'searching' || p.state === 'candidate') && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className={label}>{c.method ?? 'Method'}</div>
              <div className="mt-1 font-mono text-xl font-semibold text-zinc-50">{formatMetric(c.metric, c.target)}</div>
            </div>
            <div>
              <div className={label}>Benchmark</div>
              <div className="mt-1 text-[13px] text-zinc-300">{benchmarkText(p)}</div>
            </div>
          </div>
          <QuoteView quote={c.quote} target={c.target} />
          {c.column_check && (
            <p className="inline-flex items-center gap-1 text-[11px] text-zinc-500">
              {c.column_check === 'ok' ? 'Table column verified' : 'Column not verified'}<Hint text={HINTS.column} />
            </p>
          )}
        </div>
      )}

      {p.state === 'candidate' && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800/60 pt-3">
          {p.protocol === 'passt' ? (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-emerald-400"><CheckCircle2 className="w-3.5 h-3.5" />Protocol confirmed by quote<Hint text={HINTS.protocol} /></span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-amber-400"><AlertCircle className="w-3.5 h-3.5" />Protocol not stated in text<Hint text={HINTS.protocol} /></span>
          )}
          <span className="flex items-center gap-4 text-[11px] text-zinc-500">
            <span><span className="font-mono text-zinc-300">{p.communityHits ?? 0}</span> community {p.communityHits === 1 ? 'note' : 'notes'}</span>
            {p.githubUrl ? (
              <a href={p.githubUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-zinc-300 hover:text-white">
                <Code className="w-3.5 h-3.5" />Code
              </a>
            ) : <span className="text-zinc-600">No code found</span>}
            <a href={`https://arxiv.org/abs/${p.id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-zinc-300 hover:text-white">
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
            <span className="w-12 text-[10px] uppercase tracking-wider text-zinc-500">{name}</span>
            <div className="relative h-4 flex-1 rounded-sm bg-zinc-900">
              {view.timeline.filter((x) => x.kind === kind).map((x, i) => (
                <div key={i} title={`${x.label}: ${(x.end - x.start).toFixed(2)}s`}
                  className={`absolute top-0 h-full rounded-sm ${kind === 'tavily' ? 'bg-zinc-300' : 'bg-zinc-600'}`}
                  style={{ left: `${((x.start - view.t0) / span) * 100}%`, width: `${Math.max(0.6, ((x.end - x.start) / span) * 100)}%` }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-2 flex justify-between pl-[60px] font-mono text-[10px] text-zinc-600"><span>0s</span><span>{fmtSec(span)}</span></div>
    </div>
  )
}

export function Discovery() {
  const [datasets, setDatasets] = useState<string[]>(DATASETS)
  const [code, setCode] = useState('')
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
        body: JSON.stringify({ datasets, access_code: code }),
      })
    } catch {
      setError('The live backend is not reachable.')
      setMode('idle')
      return
    }
    if (!res.ok) {
      setError(res.status === 403 ? 'Wrong access code.'
        : res.status === 429 ? 'A run is already active or the daily limit is reached. Try again later.'
        : res.status === 404 ? 'The live backend is not available on this site.'
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
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Live Paper Discovery</h1>
        <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-zinc-400">
          Tavily searches the web for new papers on benchmarks we can re-run, reads only the protocol and results sections, and proposes checkable claims.
        </p>
      </div>

      <div className="mb-6 rounded-xl border border-zinc-800 bg-zinc-950/50 p-5">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
          <div>
            <div className={label}>Benchmarks</div>
            <div className="mt-2 flex gap-4">
              {DATASETS.map((d) => (
                <label key={d} className="inline-flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
                  <input type="checkbox" checked={datasets.includes(d)} onChange={() => toggle(d)} disabled={running} className="accent-zinc-300" />
                  {d}
                </label>
              ))}
            </div>
          </div>
          <div>
            <label htmlFor="access-code" className={label}>Access code</label>
            <input id="access-code" type="password" autoComplete="off" value={code} onChange={(e) => setCode(e.target.value)} disabled={running}
              className="mt-2 block w-44 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 font-mono text-sm text-zinc-100 outline-none focus:border-zinc-600" />
          </div>
          <div className="flex flex-wrap gap-3">
            <button onClick={startLive} disabled={running || !datasets.length || !code} className={`${btn} border-transparent bg-zinc-100 text-zinc-950 hover:bg-white`}>
              <Play className="w-4 h-4" />Start live run
            </button>
            <button onClick={startReplay} disabled={running} className={`${btn} border-zinc-800 text-zinc-300 hover:bg-zinc-800`}>
              <RotateCcw className="w-4 h-4" />Replay recorded run
            </button>
          </div>
          <div>
            <div className={label}>Replay speed</div>
            <div className="mt-2 flex gap-1">
              {[1, 4].map((s) => (
                <button key={s} onClick={() => setSpeed(s)}
                  className={`px-3 py-1.5 font-mono text-xs rounded-md border cursor-pointer ${speed === s ? 'border-zinc-600 text-zinc-100' : 'border-zinc-800 text-zinc-500 hover:text-zinc-300'}`}>{s}x</button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-zinc-800/60 pt-4 text-[12px] text-zinc-500">
          <span className={`inline-flex items-center gap-1.5 font-medium uppercase tracking-wider ${mode === 'live' ? 'text-emerald-400' : 'text-zinc-300'}`}>
            <Radio className="w-3.5 h-3.5" />{status}
          </span>
          <span>Runtime <span className="font-mono text-zinc-300">{fmtSec(view.tEnd - view.t0)}</span></span>
          <span>Papers <span className="font-mono text-zinc-300">{view.papers.length}</span></span>
          <span>Candidates <span className="font-mono text-zinc-300">{view.funnel[view.funnel.length - 1].value}</span></span>
          {recorded && events.length > 0 && <span>Recorded run from {fmtDate(events[0].ts)}</span>}
        </div>

        {error && (
          <div className="mt-4 flex flex-wrap items-center gap-4 text-[13px] text-amber-400">
            <span className="inline-flex items-center gap-2"><AlertCircle className="w-4 h-4" />{error}</span>
            <button onClick={startReplay} className="text-zinc-200 underline underline-offset-4 cursor-pointer hover:text-white">Show recorded run instead</button>
          </div>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2 items-start">
        <Section title="Tavily research feed" hint="live API calls">
          <div className="mb-4 flex flex-wrap gap-x-6 gap-y-1 text-[12px] text-zinc-500">
            <span>Tavily calls <span className="font-mono text-zinc-300">{view.tavilyCalls}</span></span>
            <span>Model calls <span className="font-mono text-zinc-300">{view.llmCalls}</span>
              {view.cachedLlmCalls > 0 && <span className="text-zinc-600"> ({view.cachedLlmCalls} cached, no cost)</span>}</span>
            <span>Model cost <span className="font-mono text-zinc-300">${view.costUsd.toFixed(3)}</span></span>
          </div>
          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {[...view.feed].reverse().map((c) => <FeedCard key={c.seq} call={c} />)}
            </AnimatePresence>
            {!view.feed.length && <p className="text-[13px] text-zinc-600">No Tavily calls yet. Start a run or replay the recorded one.</p>}
          </div>
        </Section>

        <div className="space-y-6">
          <Section title="Pipeline">
            <Funnel funnel={view.funnel} />
          </Section>
          <Section title="Papers" hint={`${papers.length} seen`}>
            <p className="mb-4 text-[12px] leading-relaxed text-zinc-500">Candidates are claims we can re-run. None of them has been re-run yet.</p>
            <div className="space-y-3">
              {papers.map((p) => <PaperCard key={p.id} p={p} active={running && view.activeId === p.id} />)}
              {!papers.length && <p className="text-[13px] text-zinc-600">Papers appear here as Tavily finds and reads them.</p>}
            </div>
          </Section>
          <Section title="Call timeline" hint="Tavily vs. model">
            {view.timeline.length ? <Timeline view={view} /> : <p className="text-[13px] text-zinc-600">The timeline fills during the run.</p>}
          </Section>
        </div>
      </div>
    </main>
  )
}

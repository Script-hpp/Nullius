import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import { ATTEMPT, RESULT, de2en } from '../labels'
import type { Event, Json } from '../data'
import { Badge, fmt } from './ui'

const STEP_MS = 650

function describe(e: Event): { title: string; body?: string; tone: 'green' | 'amber' | 'red' | 'gray' | 'violet' } {
  switch (e.type) {
    case 'run_start': return { title: `Run starts: ${e.title ?? e.paper_id}`, body: `Tavily ${e.use_tavily ? 'on' : 'off'} · setup details ${e.use_setup_details ? (e.paper_only ? 'from paper only' : 'from paper + repo') : 'off'} · ${e.seeds} runs`, tone: 'gray' }
    case 'source': return { title: `Source read (${e.kind}): ${e.mode === 'live' ? 'live Tavily call' : 'stored copy'}`, body: `${String(e.url).slice(0, 80)} · sha256 ${String(e.sha256 ?? '').slice(0, 10)}`, tone: e.mode === 'live' ? 'green' : 'gray' }
    case 'tavily_call': return { title: `Tavily role ${e.role}: ${e.endpoint}`, body: `${String(e.target).slice(0, 90)} → ${e.n_results} result(s)`, tone: 'amber' }
    case 'target_resolved': return { title: `Target value ${e.value} verified`, body: `“${String(e.quote).slice(0, 110)}” · ${e.table} · column check: ${e.column_check}`, tone: 'amber' }
    case 'setup_details': return { title: `${e.n} setup details found (each with a verbatim quote)`, body: (e.details ?? []).slice(0, 4).map((d: Json) => `${d.topic}: ${String(d.value).slice(0, 30)}`).join(' · '), tone: 'amber' }
    case 'claim_start': return { title: `Claim: ${e.text}`, body: `target ${e.target} ± ${e.tolerance}`, tone: 'gray' }
    case 'plan': return { title: 'Research plan written (reasoning model)', body: String(e.text ?? '').slice(0, 160), tone: 'violet' }
    case 'code_written': return { title: `Experiment code written (${e.code_hash})`, tone: 'violet' }
    case 'attempt': { const s = de2en(ATTEMPT, e.status); return { title: `Branch ${String(e.branch_id).slice(-8)} · seed ${e.seed}: ${s.label}`, body: e.measured != null ? `measured ${fmt(e.measured)}` : String(e.error ?? '').split('\n').filter(Boolean).slice(-1)[0], tone: s.tone } }
    case 'repair': return { title: 'Repair: rollback to snapshot, new code', body: String(e.error ?? '').split('\n').filter(Boolean).slice(-1)[0], tone: 'red' }
    case 'ablation': return { title: `Graph ablation: ${e.verdict}`, body: `${fmt(e.accuracy_full)} with graph → ${fmt(e.accuracy_ablated)} without edges`, tone: e.verdict === 'Graph genutzt' ? 'green' : 'gray' }
    case 'verdict': { const r = de2en(RESULT, e.urteil); return { title: `Verdict: ${r.label}`, body: `mean ${fmt(e.mean)} (std ${fmt(e.std)})`, tone: r.tone } }
    case 'run_end': return { title: 'Run finished', body: `cost $${e.cost_net_usd} net / $${e.cost_gross_usd} gross`, tone: 'gray' }
    default: return { title: e.type, tone: 'gray' }
  }
}

export function Replay({ events }: { events: Event[] }) {
  // Viele parallele Seed-Läufe: nur die ersten 12 Läufe einzeln zeigen, Rest als Zusammenfassung.
  const shown = useMemo(() => {
    const out: Event[] = []
    let attempts = 0
    for (const e of events) {
      if (e.type === 'attempt') { attempts++; if (attempts > 12) continue }
      out.push(e)
    }
    return { list: out, hidden: attempts > 12 ? attempts - 12 : 0 }
  }, [events])
  const list = shown.list
  const [i, setI] = useState(list.length)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { setI(list.length); setPlaying(false) }, [list])
  useEffect(() => {
    if (!playing) return
    if (i >= list.length) { setPlaying(false); return }
    const t = setTimeout(() => setI((x) => x + 1), STEP_MS)
    return () => clearTimeout(t)
  }, [playing, i, list.length])
  const visible = list.slice(0, i)
  const tones = { green: 'var(--green)', amber: 'var(--amber)', red: 'var(--red)', gray: 'var(--muted)', violet: 'var(--violet)' }

  return (
    <div>
      <div className="mb-3 flex items-center gap-3">
        <button className="rounded-lg px-3 py-1.5 text-sm font-medium" style={{ background: 'var(--blue)', color: '#04101d' }}
          onClick={() => { if (i >= list.length) setI(0); setPlaying((p) => !p) }}>
          {playing ? 'Pause' : i >= list.length ? '↻ Replay run' : '▶ Continue'}
        </button>
        <button className="panel-2 px-3 py-1.5 text-sm" onClick={() => { setPlaying(false); setI((x) => Math.min(list.length, x + 1)) }}>Step</button>
        <button className="panel-2 px-3 py-1.5 text-sm" onClick={() => { setPlaying(false); setI(list.length) }}>Show all</button>
        <div className="muted ml-auto text-xs">{i} / {list.length} events{shown.hidden ? ` · ${shown.hidden} further runs not listed` : ''}</div>
      </div>
      <div className="mb-3 h-1 overflow-hidden rounded-full" style={{ background: 'var(--line)' }}>
        <motion.div className="h-full" style={{ background: 'var(--blue)' }} animate={{ width: `${list.length ? (i / list.length) * 100 : 0}%` }} transition={{ duration: 0.3 }} />
      </div>
      <ol className="space-y-2">
        <AnimatePresence initial={false}>
          {visible.map((e, k) => {
            const d = describe(e)
            return (
              <motion.li key={k} layout initial={{ opacity: 0, y: -10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.25 }}
                className="panel-2 flex items-start gap-3 px-3 py-2" style={{ borderLeft: `3px solid ${tones[d.tone]}` }}>
                <span className="muted mono w-6 shrink-0 text-right text-xs">{k + 1}</span>
                <div className="min-w-0 text-sm">
                  <div className="font-medium">{d.title}</div>
                  {d.body && <div className="muted mono break-words text-xs">{d.body}</div>}
                </div>
                {e.type === 'verdict' && <span className="ml-auto"><Badge tone={d.tone}>{de2en(RESULT, e.urteil).label}</Badge></span>}
              </motion.li>
            )
          })}
        </AnimatePresence>
      </ol>
    </div>
  )
}

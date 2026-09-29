import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import { ATTEMPT, RESULT, de2en } from '../labels'
import type { Event, Json } from '../data'
import { fmt } from './ui'
import { Play, Pause } from 'lucide-react'

const STEP_MS = 650

function describe(e: Event): { title: string; body?: string; tone: 'green' | 'amber' | 'red' | 'gray' | 'violet' } {
  switch (e.type) {
    case 'run_start': return { title: `Evaluation started: ${e.title ?? e.paper_id}`, body: `Web Search: ${e.use_tavily ? 'Enabled' : 'Disabled'} · Setup Details: ${e.use_setup_details ? (e.paper_only ? 'Paper only' : 'Paper + Code repo') : 'Off'} · Parallel Runs: ${e.seeds}`, tone: 'gray' }
    case 'source': return { title: `Source read (${e.kind === 'pdf' ? 'PDF' : 'Web'}): ${e.mode === 'live' ? 'Live Search' : 'Cached Data'}`, body: `${String(e.url).slice(0, 80)}`, tone: e.mode === 'live' ? 'green' : 'gray' }
    case 'tavily_call': return { title: `Search Query Executed`, body: `${String(e.target).slice(0, 90)} → Found ${e.n_results} result(s)`, tone: 'amber' }
    case 'target_resolved': return { title: `Target value ${e.value} successfully extracted`, body: `Quote: "${String(e.quote).slice(0, 110)}"`, tone: 'green' }
    case 'setup_details': return { title: `Extracted ${e.n} setup configurations from paper`, body: (e.details ?? []).slice(0, 3).map((d: Json) => `${d.topic}`).join(' · '), tone: 'amber' }
    case 'claim_start': return { title: `Analyzing Claim: ${e.text.replace(/erreicht/g, 'achieves').replace(/Accuracy auf/g, 'accuracy on').replace(/groesste Komponente/g, 'largest component').replace(/zufaellige/g, 'random')}`, body: `Expected Result: ${e.target} (Tolerance: ±${e.tolerance})`, tone: 'gray' }
    case 'plan': return { title: 'Execution plan formulated', body: String(e.text ?? '').slice(0, 160) + '...', tone: 'violet' }
    case 'code_written': return { title: `Sandbox code generated successfully`, tone: 'violet' }
    case 'attempt': { const s = de2en(ATTEMPT, e.status); return { title: `Sandbox run (Seed ${e.seed}) completed`, body: e.measured != null ? `Result: ${fmt(e.measured)}` : 'Execution failed', tone: s.tone } }
    case 'repair': return { title: 'Self-healing triggered (Syntax/Runtime error)', body: 'Agent generated a patch and restarted the sandbox', tone: 'red' }
    case 'ablation': return { title: `Method validation check`, body: `Accuracy with method: ${fmt(e.accuracy_full)} → without method: ${fmt(e.accuracy_ablated)}`, tone: e.verdict === 'Graph genutzt' ? 'green' : 'gray' }
    case 'verdict': { const r = de2en(RESULT, e.urteil); return { title: `Final verdict reached`, body: `Calculated mean: ${fmt(e.mean)}`, tone: r.tone } }
    case 'run_end': return { title: 'Evaluation finished', body: `Compute Cost: $${e.cost_net_usd}`, tone: 'gray' }
    default: return { title: e.type.replace(/_/g, ' '), tone: 'gray' }
  }
}

export function Replay({ events }: { events: Event[] }) {
  const shown = useMemo(() => {
    const out: Event[] = []
    let attempts = 0
    for (const e of events) {
      if (e.type === 'attempt') { attempts++; if (attempts > 6) continue } // Only show first 6 attempts in log to avoid clutter
      out.push(e)
    }
    return { list: out, hidden: attempts > 6 ? attempts - 6 : 0 }
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
  const tones = { green: 'bg-emerald-500', amber: 'bg-amber-500', red: 'bg-rose-500', gray: 'bg-zinc-700', violet: 'bg-violet-500' }

  return (
    <div className="bg-[#0a0a0a] border border-zinc-800/80 rounded-xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 rounded-lg bg-zinc-100 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-950 transition-colors hover:bg-white"
            onClick={() => { if (i >= list.length) setI(0); setPlaying((p) => !p) }}>
            {playing ? <><Pause className="w-3.5 h-3.5" /> Pause</> : i >= list.length ? <><Play className="w-3.5 h-3.5" /> Replay</> : <><Play className="w-3.5 h-3.5" /> Resume</>}
          </button>
          <button className="text-xs uppercase tracking-wider font-medium text-zinc-500 hover:text-zinc-300 transition-colors px-2" onClick={() => { setPlaying(false); setI(list.length) }}>Show All</button>
        </div>
        <div className="text-zinc-500 text-[11px] uppercase tracking-wider font-mono">
          Step {i} of {list.length}
        </div>
      </div>
      
      <div className="mb-8 h-1 w-full overflow-hidden rounded-full bg-zinc-900">
        <motion.div className="h-full bg-zinc-400" animate={{ width: `${list.length ? (i / list.length) * 100 : 0}%` }} transition={{ duration: 0.3 }} />
      </div>
      
      <div className="relative border-l border-zinc-800/80 ml-3 space-y-6 pb-4">
        <AnimatePresence initial={false}>
          {visible.map((e, k) => {
            const d = describe(e)
            return (
              <motion.div key={k} layout initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.25 }}
                className="relative pl-6 group">
                <div className={`absolute -left-[5px] top-1.5 h-2 w-2 rounded-full ring-4 ring-[#0a0a0a] ${tones[d.tone]}`} />
                <div className="text-sm font-medium text-zinc-200">{d.title}</div>
                {d.body && <div className="mt-1 text-[13px] text-zinc-500 leading-relaxed">{d.body}</div>}
              </motion.div>
            )
          })}
        </AnimatePresence>
        {shown.hidden > 0 && i === list.length && (
          <div className="relative pl-6">
            <div className="absolute -left-[5px] top-1.5 h-2 w-2 rounded-full ring-4 ring-[#0a0a0a] bg-zinc-700" />
            <div className="text-sm font-medium text-zinc-500 italic">...and {shown.hidden} more parallel sandbox executions.</div>
          </div>
        )}
      </div>
    </div>
  )
}

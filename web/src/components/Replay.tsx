import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import { ATTEMPT, RESULT, de2en } from '../labels'
import type { Event } from '../data'
import { toPercent } from './ui'
import { Play, Pause } from 'lucide-react'

const STEP_MS = 650

function describe(e: Event): { title: string; body?: string; tone: 'green' | 'amber' | 'red' | 'gray' | 'violet' } {
  switch (e.type) {
    case 'run_start': return { title: `Evaluation started: ${e.title ?? e.paper_id}`, body: `Web Search: ${e.use_tavily ? 'Enabled' : 'Disabled'} · Parallel Runs: ${e.seeds}`, tone: 'gray' }
    case 'source': return { title: `Paper content fetched (${e.kind === 'pdf' ? 'PDF' : 'Web'})`, body: `Source: ${String(e.url).slice(0, 80)}`, tone: e.mode === 'live' ? 'green' : 'gray' }
    case 'tavily_call': return { title: `Search Query Executed`, body: `Found ${e.n_results} relevant documents`, tone: 'amber' }
    case 'target_resolved': return { title: `Target metric ${toPercent(e.value)} successfully extracted`, body: `Source: ${e.table || 'Text'}`, tone: 'green' }
    case 'setup_details': return { title: `Extracted ${e.n} setup configurations from paper`, body: `Hyperparameters, dataset splits, epochs, etc.`, tone: 'amber' }
    case 'claim_start': return { title: `Analyzing Claim`, body: `Expected Result: ${toPercent(e.target)} (Tolerance: ±${toPercent(e.tolerance)})`, tone: 'gray' }
    case 'plan': return { title: 'Execution plan formulated', body: 'Reasoning model prepared sandbox strategy', tone: 'violet' }
    case 'code_written': return { title: `Sandbox code generated successfully`, tone: 'violet' }
    case 'attempt': { const s = de2en(ATTEMPT, e.status); return { title: `Sandbox run (Seed ${e.seed}) completed`, body: e.measured != null ? `Result: ${toPercent(e.measured)}` : 'Execution failed', tone: s.tone } }
    case 'repair': return { title: 'Self-healing triggered (Syntax/Runtime error)', body: 'Agent generated a patch and restarted the sandbox', tone: 'red' }
    case 'ablation': return { title: `Method validation check`, body: `Accuracy with method: ${toPercent(e.accuracy_full)} → without method: ${toPercent(e.accuracy_ablated)}`, tone: e.verdict === 'Graph genutzt' ? 'green' : 'gray' }
    case 'verdict': { const r = de2en(RESULT, e.urteil); return { title: `Final verdict reached`, body: `Calculated mean: ${toPercent(e.mean)}`, tone: r.tone } }
    case 'run_end': return { title: 'Evaluation finished', body: `Compute Cost: $${e.cost_net_usd}`, tone: 'gray' }
    default: return { title: e.type.replace(/_/g, ' '), tone: 'gray' }
  }
}

export function Replay({ events }: { events: Event[] }) {
  const shown = useMemo(() => {
    const out: Event[] = []
    let attempts = 0
    for (const e of events) {
      if (e.type === 'attempt') { attempts++; if (attempts > 6) continue }
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
  const tones = { green: 'bg-tone-green-solid', amber: 'bg-tone-amber-solid', red: 'bg-tone-red-solid', gray: 'bg-dot', violet: 'bg-tone-violet-solid' }

  return (
    <div className="bg-surface border border-line/80 rounded-xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 rounded-lg bg-inverse px-4 py-2 text-xs font-semibold uppercase tracking-wider text-on-inverse transition-colors hover:bg-inverse-hover cursor-pointer"
            onClick={() => { if (i >= list.length) setI(0); setPlaying((p) => !p) }}>
            {playing ? <><Pause className="w-3.5 h-3.5" /> Pause</> : i >= list.length ? <><Play className="w-3.5 h-3.5" /> Replay</> : <><Play className="w-3.5 h-3.5" /> Resume</>}
          </button>
          <button className="text-xs uppercase tracking-wider font-medium text-muted hover:text-fg-3 transition-colors px-2 cursor-pointer" onClick={() => { setPlaying(false); setI(list.length) }}>Show All</button>
        </div>
        <div className="text-muted text-[11px] uppercase tracking-wider font-mono">
          Step {i} of {list.length}
        </div>
      </div>
      
      <div className="mb-8 h-1 w-full overflow-hidden rounded-full bg-raised">
        <motion.div className="h-full bg-fill-mid" animate={{ width: `${list.length ? (i / list.length) * 100 : 0}%` }} transition={{ duration: 0.3 }} />
      </div>
      
      <div className="relative border-l border-line/80 ml-3 space-y-6 pb-4">
        <AnimatePresence initial={false}>
          {visible.map((e, k) => {
            const d = describe(e)
            return (
              <motion.div key={k} layout initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.25 }}
                className="relative pl-6 group">
                <div className={`absolute -left-[5px] top-1.5 h-2 w-2 rounded-full ring-4 ring-surface ${tones[d.tone]}`} />
                <div className="text-sm font-medium text-fg-2">{d.title}</div>
                {d.body && <div className="mt-1 text-xs text-fg-4 leading-relaxed">{d.body}</div>}
              </motion.div>
            )
          })}
        </AnimatePresence>
        {shown.hidden > 0 && i === list.length && (
          <div className="relative pl-6">
            <div className="absolute -left-[5px] top-1.5 h-2 w-2 rounded-full ring-4 ring-surface bg-dot" />
            <div className="text-sm font-medium text-muted italic">...and {shown.hidden} more parallel sandbox executions.</div>
          </div>
        )}
      </div>
    </div>
  )
}

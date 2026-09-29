import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Layout } from './components/Layout'
import { AttemptChart } from './components/Gauge'
import { BranchTree } from './components/BranchTree'
import { EvidenceSplit } from './components/Evidence'
import { Replay } from './components/Replay'
import { Badge, Section, Stat, fmt } from './components/ui'
import { loadIndex, loadRun, type Json, type RunData, type RunRef } from './data'
import { METHOD, OVERALL, RESULT, de2en } from './labels'
import { ArrowRight, BookOpen, Activity, PlayCircle, BarChart2, CheckCircle2, AlertCircle } from 'lucide-react'

const useHash = () => {
  const [h, setH] = useState(location.hash.slice(1))
  useEffect(() => {
    const f = () => setH(location.hash.slice(1))
    addEventListener('hashchange', f)
    return () => removeEventListener('hashchange', f)
  }, [])
  return h
}

function parseDate(ts: string) {
  const dateStr = ts.split('-')[0]
  if (dateStr.length !== 8) return ts
  const d = new Date(parseInt(dateStr.slice(0,4)), parseInt(dateStr.slice(4,6))-1, parseInt(dateStr.slice(6,8)))
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function DashboardHeader({ count }: { count: number }) {
  return (
    <div className="mb-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Overview</h1>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-transparent border border-zinc-800 text-zinc-300 text-sm font-medium rounded-lg hover:bg-zinc-800 transition-colors shadow-sm cursor-pointer">
            Filter runs
          </button>
          <button className="px-4 py-2 bg-zinc-100 text-zinc-950 border border-transparent text-sm font-medium rounded-lg hover:bg-white hover:opacity-90 transition-all shadow-sm flex items-center gap-2 cursor-pointer">
            <PlayCircle className="w-4 h-4" />
            New Evaluation
          </button>
        </div>
      </div>
      
      {/* Quick Stats Row */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-5 flex flex-col gap-3 rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900 to-zinc-950 shadow-sm">
          <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium uppercase tracking-wider">
            <div className="p-1.5 rounded-md bg-zinc-800/50 text-zinc-400 border border-zinc-700/50">
              <Activity className="w-3.5 h-3.5" />
            </div>
            Total Runs
          </div>
          <span className="font-mono text-2xl font-semibold text-zinc-100">{count}</span>
        </div>
        <div className="p-5 flex flex-col gap-3 rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900 to-zinc-950 shadow-sm">
          <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium uppercase tracking-wider">
            <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <CheckCircle2 className="w-3.5 h-3.5" />
            </div>
            Reproduced
          </div>
          <span className="font-mono text-2xl font-semibold text-zinc-100">2</span>
        </div>
        <div className="p-5 flex flex-col gap-3 rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900 to-zinc-950 shadow-sm">
          <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium uppercase tracking-wider">
            <div className="p-1.5 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <AlertCircle className="w-3.5 h-3.5" />
            </div>
            Partially
          </div>
          <span className="font-mono text-2xl font-semibold text-zinc-100">2</span>
        </div>
        <div className="p-5 flex flex-col gap-3 rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900 to-zinc-950 shadow-sm">
          <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium uppercase tracking-wider">
            <div className="p-1.5 rounded-md bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <BarChart2 className="w-3.5 h-3.5" />
            </div>
            Datasets
          </div>
          <span className="font-mono text-2xl font-semibold text-zinc-100">3</span>
        </div>
      </div>
    </div>
  )
}

function Home({ runs }: { runs: RunRef[] }) {
  const [data, setData] = useState<Record<string, RunData>>({})
  useEffect(() => { runs.forEach((r) => loadRun(r).then((d) => setData((x) => ({ ...x, [r.id]: d })))) }, [runs])
  return (
    <main className="mx-auto max-w-6xl px-8 pb-16">
      <DashboardHeader count={runs.length} />
      
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-zinc-500">Recent Evaluations</h2>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {runs.map((r, i) => {
          const d = data[r.id]
          const v = d?.dossier?.verdicts?.[0]
          const claim = d?.dossier?.claims?.[0]
          const res = v ? de2en(RESULT, v.urteil) : null
          
          return (
            <motion.a key={r.id} href={`#/run/${r.id}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className="flex flex-col p-6 rounded-xl border border-zinc-800 bg-zinc-950/50 no-underline transition-all duration-200 hover:border-zinc-700 hover:bg-zinc-900/80 group">
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium font-mono">
                  <BookOpen className="w-3.5 h-3.5" />
                  {r.paper_id}
                </div>
                <div className="text-xs text-zinc-500">{parseDate(r.timestamp)}</div>
              </div>
              
              <h3 className="mt-1 text-lg font-semibold leading-snug text-zinc-100 group-hover:text-white transition-colors">{r.title}</h3>
              {claim && <p className="mt-2 text-sm text-zinc-400 line-clamp-2 leading-relaxed">{claim.text}</p>}
              
              <div className="mt-auto pt-6 flex items-center justify-between">
                {v ? (
                  <div className="flex flex-wrap items-center gap-3">
                    <Badge tone={res!.tone}>{res!.label}</Badge>
                    <span className="font-mono flex items-baseline gap-1.5">
                      <span className="text-emerald-400 font-bold">{fmt(v.measured_mean, 3)}</span>
                      <span className="text-zinc-600 text-xs">vs</span>
                      <span className="text-zinc-500 text-xs">{claim?.target}</span>
                    </span>
                  </div>
                ) : (
                  <span className="text-xs text-zinc-500">Evaluating...</span>
                )}
                
                <div className="text-zinc-600 group-hover:text-zinc-300 transition-all duration-200 group-hover:translate-x-1">
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </motion.a>
          )
        })}
      </div>
    </main>
  )
}

function RunView({ ref_ }: { ref_: RunRef }) {
  const [run, setRun] = useState<RunData | null>(null)
  const [selected, setSelected] = useState<Json | null>(null)
  const [tab, setTab] = useState<'code' | 'zeugnis' | 'plan'>('zeugnis')
  useEffect(() => { loadRun(ref_).then(setRun) }, [ref_])
  
  if (!run) return <main className="mx-auto max-w-6xl px-8 py-10 muted">Loading run data…</main>
  const d = run.dossier
  const claim: Json = d.claims?.[0]
  const v: Json = d.verdicts?.[0]
  if (!claim) return <main className="mx-auto max-w-6xl px-8 py-10 muted">Empty run.</main>
  
  const res = v ? de2en(RESULT, v.urteil) : { label: 'no verdict', tone: 'gray' as const }
  const meth = v ? de2en(METHOD, v.methode) : { label: '–', tone: 'gray' as const }
  const attempts: Json[] = (d.attempts ?? []).filter((a: Json) => a.claim_id === claim.id)
  const end = run.events.find((e) => e.type === 'run_end')
  const cost = end?.cost_net_usd ?? attempts[attempts.length - 1]?.cost_usd
  const paperUrl = claim.target_evidence?.url
  const tabs = { zeugnis: run.zeugnis, code: run.code, plan: run.plan }

  return (
    <main className="mx-auto max-w-6xl space-y-8 px-8 pb-20">
      <div className="pt-2">
        <a href="#" className="inline-flex items-center gap-1.5 text-zinc-500 text-sm font-medium hover:text-zinc-200 transition-colors">
          <ArrowRight className="w-4 h-4 rotate-180" /> Back to Overview
        </a>
      </div>
      
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="pb-8 border-b border-zinc-800">
        <div className="flex items-center gap-3 text-zinc-500 text-xs mb-3 font-mono">
          <span>{run.ref.paper_id}</span>
          <span>•</span>
          <span>{parseDate(run.ref.timestamp)}</span>
          <span>•</span>
          <span className="uppercase tracking-wider">{d.mode === 'nachbau' ? 'rebuild (own code)' : d.mode}</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-zinc-100 leading-tight">{run.ref.title}</h1>
        <p className="mt-4 text-lg text-zinc-400 max-w-3xl leading-relaxed">{claim.text}</p>
        
        {paperUrl && (
          <a className="inline-flex items-center gap-2 mt-4 text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors" href={paperUrl} target="_blank" rel="noreferrer">
            <BookOpen className="w-4 h-4" />
            View Source Paper
          </a>
        )}
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Section title="Final Verdict" hint="Evaluation result" className="lg:col-span-1 border-t-4" style={{ borderTopColor: `var(--${res.tone})` }}>
          <div className="space-y-4">
            <div>
              <div className="text-zinc-500 mb-2 text-xs font-semibold uppercase tracking-wider">1 · Reproducibility</div>
              <Badge tone={res.tone}>{res.label}</Badge>
            </div>
            <div>
              <div className="text-zinc-500 mb-2 text-xs font-semibold uppercase tracking-wider">2 · Method fidelity</div>
              <Badge tone={meth.tone}>{meth.label}</Badge>
            </div>
            <div className="pt-2">
              <div className="text-zinc-500 mb-1 text-xs font-semibold uppercase tracking-wider">Overall Statement</div>
              <span className="text-sm font-medium text-zinc-200">{OVERALL[v?.gesamt ?? ''] ?? (v?.methode === 'nicht geprueft' ? 'provisional' : '–')}</span>
            </div>
            <p className="text-zinc-500 text-xs leading-relaxed pt-3 border-t border-zinc-800 mt-4">
              A hit on the number alone proves nothing about the method: a closed-form regression can hit a target value, so the second axis exists.
            </p>
          </div>
          <div className="mt-6 grid grid-cols-2 gap-4 bg-zinc-950/50 p-4 rounded-xl border border-zinc-800/50">
            <Stat label="measured" value={fmt(v?.measured_mean, 4)} sub={`± ${fmt(v?.measured_std, 4)} std`} />
            <Stat label="target" value={claim.target} sub={`± ${claim.tolerance}`} />
          </div>
          <div className="mt-4 text-right">
             <span className="text-xs text-zinc-500 font-mono">Cost: {cost != null ? `$${Number(cost).toFixed(3)} net` : '–'}</span>
          </div>
        </Section>
        <Section title="Measurements" hint={`${attempts.length} runs, one bar each`} className="lg:col-span-2">
          <AttemptChart dossier={d} claimId={claim.id} />
        </Section>
      </div>

      <Section title="Evidence" hint="left: what Tavily found · right: what the sandbox measured">
        <EvidenceSplit claimId={claim.id} dossier={d} events={run.events} selected={selected} />
      </Section>

      <Section title="Branch Tree" hint="click a run for details · violet edges = repair after an error">
        <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />
      </Section>

      <Section title="Run Log" hint="Live event stream">
        <Replay events={run.events} />
      </Section>

      <Section title="Source Files">
        <div className="mb-4 flex gap-2 border-b border-zinc-800 pb-3">
          {(['zeugnis', 'code', 'plan'] as const).map((t) => tabs[t] != null && (
            <button key={t} onClick={() => setTab(t)} 
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors cursor-pointer ${tab === t ? 'bg-zinc-100 text-zinc-900 shadow-sm' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'}`}>
              {t === 'zeugnis' ? 'Certificate' : t === 'code' ? 'Agent Code' : 'Execution Plan'}
            </button>
          ))}
        </div>
        <pre className="font-mono max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-xl p-5 bg-zinc-950 border border-zinc-800 text-zinc-300 text-[13px] leading-relaxed">
          {tabs[tab] ?? 'not available'}
        </pre>
      </Section>
    </main>
  )
}


export default function App() {
  const hash = useHash()
  const [runs, setRuns] = useState<RunRef[] | null>(null)
  useEffect(() => { loadIndex().then(setRuns) }, [])
  const m = hash.match(/^\/run\/(.+)$/)
  const ref = m && runs?.find((r) => r.id === decodeURIComponent(m[1]))
  
  return (
    <Layout currentHash={hash}>
      <div className="py-6 min-h-screen bg-black">
        {runs === null ? (
          <div className="mx-auto max-w-6xl px-8 py-20 flex flex-col items-center justify-center gap-4">
            <Activity className="w-8 h-8 text-zinc-600 animate-pulse" />
            <span className="text-zinc-500 font-medium">Loading evaluation data...</span>
          </div>
        ) : ref ? <RunView ref_={ref} key={ref.id} /> : <Home runs={runs} />}
      </div>
    </Layout>
  )
}

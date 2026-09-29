import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Layout } from './components/Layout'
import { AttemptChart } from './components/Gauge'
import { BranchTree } from './components/BranchTree'
import { EvidenceSplit } from './components/Evidence'
import { Replay } from './components/Replay'
import { Badge, Section, Stat, } from './components/ui'
import { loadIndex, loadRun, type Json, type RunData, type RunRef } from './data'
import { METHOD, OVERALL, RESULT, de2en, type Tone } from './labels'
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

function translateClaim(text: string) {
  if (!text) return text
  let t = text
  t = t.replace(/erreicht/g, 'achieves')
  t = t.replace(/Accuracy auf/g, 'accuracy on')
  t = t.replace(/groesste Komponente/g, 'largest component')
  t = t.replace(/zufaellige/g, 'random')
  t = t.replace(/,(\d) %/g, '.$1%') // "81,0 %" -> "81.0%"
  return t
}

function colorForTone(tone: Tone) {
  if (tone === 'green') return 'text-emerald-400'
  if (tone === 'amber') return 'text-amber-400'
  if (tone === 'red') return 'text-rose-400'
  if (tone === 'violet') return 'text-violet-400'
  return 'text-zinc-400'
}

function toPercent(val: string | number | null | undefined): string {
  if (val === null || val === undefined || val === '') return '–'
  const n = Number(val)
  if (Number.isNaN(n)) return String(val)
  // If the number is likely a fraction (<= 1.0 or similar), convert to percentage
  // Since the user asked to convert ALL decimals like 0.81 to 81.0%, we assume they are all fractions if < 2
  if (n <= 2.0 && n > 0) {
    return (n * 100).toFixed(1) + '%'
  }
  return n.toFixed(1) + '%'
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
            Verified
          </div>
          <span className="font-mono text-2xl font-semibold text-zinc-100">2</span>
        </div>
        <div className="p-5 flex flex-col gap-3 rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900 to-zinc-950 shadow-sm">
          <div className="flex items-center gap-2 text-zinc-400 text-xs font-medium uppercase tracking-wider">
            <div className="p-1.5 rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20">
              <AlertCircle className="w-3.5 h-3.5" />
            </div>
            Partial
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
        <h2 className="text-[11px] font-semibold tracking-widest uppercase text-zinc-500">Recent Evaluations</h2>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {runs.map((r, i) => {
          const d = data[r.id]
          const v = d?.dossier?.verdicts?.[0]
          const claim = d?.dossier?.claims?.[0]
          const res = v ? de2en(RESULT, v.urteil) : null
          
          return (
            <motion.a key={r.id} href={`#/run/${r.id}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className="flex flex-col p-6 rounded-xl border border-zinc-800 bg-zinc-950/50 no-underline transition-all duration-200 hover:border-zinc-700 hover:bg-zinc-900/80 group cursor-pointer block">
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-2 text-zinc-400 text-[11px] font-medium font-mono uppercase tracking-wider">
                  <BookOpen className="w-3.5 h-3.5" />
                  {r.paper_id}
                </div>
                <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{parseDate(r.timestamp)}</div>
              </div>
              
              <h3 className="text-lg font-semibold leading-snug text-zinc-100 group-hover:text-white transition-colors">{r.title}</h3>
              {claim && <p className="mt-2 text-[13px] text-zinc-400 line-clamp-2 leading-relaxed mb-6">{translateClaim(claim.text)}</p>}
              
              <div className="mt-auto border-t border-zinc-800/50 pt-5">
                {v ? (
                  <div className="flex flex-col gap-4">
                    <Badge tone={res!.tone}>{res!.label}</Badge>
                    
                    <div className="flex flex-col gap-2 w-full bg-[#0a0a0a] rounded-lg p-3 border border-zinc-800/50">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-zinc-500 uppercase tracking-widest font-semibold text-[10px]">Claimed</span>
                        <span className="font-mono text-zinc-400">{toPercent(claim?.target)}</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-zinc-500 uppercase tracking-widest font-semibold text-[10px]">Measured</span>
                        <span className={`font-mono font-bold ${colorForTone(res!.tone)}`}>{toPercent(v.measured_mean)}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <span className="text-xs text-zinc-500 uppercase tracking-wider font-medium">Evaluating...</span>
                )}
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
  
  const res = v ? de2en(RESULT, v.urteil) : { label: 'NO VERDICT', tone: 'gray' as const }
  const meth = v ? de2en(METHOD, v.methode) : { label: '–', tone: 'gray' as const }
  const attempts: Json[] = (d.attempts ?? []).filter((a: Json) => a.claim_id === claim.id)
  const end = run.events.find((e) => e.type === 'run_end')
  const cost = end?.cost_net_usd ?? attempts[attempts.length - 1]?.cost_usd
  const paperUrl = claim.target_evidence?.url
  const tabs = { zeugnis: run.zeugnis, code: run.code, plan: run.plan }

  return (
    <main className="mx-auto max-w-6xl space-y-8 px-8 pb-20">
      <div className="pt-2">
        <a href="#" className="inline-flex items-center gap-1.5 text-zinc-500 text-xs font-medium uppercase tracking-wider hover:text-zinc-200 transition-colors">
          <ArrowRight className="w-3.5 h-3.5 rotate-180" /> Back to Overview
        </a>
      </div>
      
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="pb-8 border-b border-zinc-800/50">
        <div className="flex items-center gap-3 text-zinc-500 text-xs mb-3 font-mono">
          <span>{run.ref.paper_id}</span>
          <span>•</span>
          <span>{parseDate(run.ref.timestamp)}</span>
          <span>•</span>
          <span className="uppercase tracking-wider">{d.mode === 'nachbau' ? 'REBUILD (OWN CODE)' : d.mode.toUpperCase()}</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-zinc-100 leading-tight">{run.ref.title}</h1>
        <p className="mt-4 text-lg text-zinc-400 max-w-3xl leading-relaxed">{translateClaim(claim.text)}</p>
        
        {paperUrl && (
          <a className="inline-flex items-center gap-2 mt-5 text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors" href={paperUrl} target="_blank" rel="noreferrer">
            <BookOpen className="w-4 h-4" />
            View Source Paper
          </a>
        )}
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Section title="Final Verdict" className="lg:col-span-1 border-t-2" style={{ borderTopColor: `var(--${res.tone})` }}>
          <div className="space-y-6">
            <div>
              <div className="text-zinc-500 mb-2 text-[10px] font-semibold uppercase tracking-widest">1 · Reproducibility</div>
              <Badge tone={res.tone}>{res.label}</Badge>
            </div>
            <div>
              <div className="text-zinc-500 mb-2 text-[10px] font-semibold uppercase tracking-widest">2 · Method Fidelity</div>
              <Badge tone={meth.tone}>{meth.label}</Badge>
            </div>
            <div className="pt-3 border-t border-zinc-800/50">
              <div className="text-zinc-500 mb-1 text-[10px] font-semibold uppercase tracking-widest">Overall Statement</div>
              <span className="text-sm font-medium text-zinc-200">{OVERALL[v?.gesamt ?? ''] ?? (v?.methode === 'nicht geprueft' ? 'Provisional' : '–')}</span>
            </div>
            <p className="text-zinc-600 text-[11px] leading-relaxed">
              A hit on the number alone proves nothing about the method: a closed-form regression can hit a target value, so the second axis exists.
            </p>
          </div>
          <div className="mt-6 grid grid-cols-2 gap-4 bg-zinc-950 p-4 rounded-xl border border-zinc-800/30">
            <Stat label="Measured" value={toPercent(v?.measured_mean)} />
            <Stat label="Claimed" value={toPercent(claim.target)} />
          </div>
          <div className="mt-4 text-right">
             <span className="text-[10px] uppercase tracking-wider text-zinc-600 font-mono">Cost: {cost != null ? `$${Number(cost).toFixed(3)} net` : '–'}</span>
          </div>
        </Section>
        <Section title="Measurements" hint={`${attempts.length} runs`} className="lg:col-span-2">
          <AttemptChart dossier={d} claimId={claim.id} />
        </Section>
      </div>

      <Section title="Evidence" hint="Left: Extracted from paper · Right: Measured in sandbox">
        <EvidenceSplit claimId={claim.id} dossier={d} events={run.events} selected={selected} />
      </Section>

      <Section title="Branch Tree" hint="Violet edges indicate agent repairs">
        <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />
      </Section>

      <Section title="Execution Log" hint="Live event stream">
        <Replay events={run.events} />
      </Section>

      <Section title="Source Files">
        <div className="mb-4 flex gap-2 border-b border-zinc-800/50 pb-3">
          {(['zeugnis', 'code', 'plan'] as const).map((t) => tabs[t] != null && (
            <button key={t} onClick={() => setTab(t)} 
              className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider rounded-lg transition-colors cursor-pointer ${tab === t ? 'bg-zinc-100 text-zinc-900 shadow-sm' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}>
              {t === 'zeugnis' ? 'Certificate' : t === 'code' ? 'Agent Code' : 'Execution Plan'}
            </button>
          ))}
        </div>
        <pre className="font-mono max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-xl p-5 bg-[#0a0a0a] border border-zinc-800/50 text-zinc-300 text-[13px] leading-relaxed">
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
            <span className="text-zinc-500 text-sm font-medium uppercase tracking-wider">Loading Evaluation Data...</span>
          </div>
        ) : ref ? <RunView ref_={ref} key={ref.id} /> : <Home runs={runs} />}
      </div>
    </Layout>
  )
}

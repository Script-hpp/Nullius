import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Layout } from './components/Layout'
import { AttemptChart } from './components/Gauge'
import { BranchTree } from './components/BranchTree'
import { EvidenceSplit } from './components/Evidence'
import { Replay } from './components/Replay'
import { Discovery } from './components/Discovery'
import { Catalog } from './components/Catalog'
import { LiveRun } from './components/LiveRun'
import { DATA_CHANGED } from './live'
import { CatalogCallout, PreliminaryChip } from './components/CatalogBits'
import { Badge, Section, toPercent } from './components/ui'
import { loadIndex, loadRun, type Json, type RunData, type RunRef } from './data'
import { NOT_ASSESSABLE, RESULT, TONE_VAR, catalogLabel, de2en, isPreliminary, type Tone } from './labels'
import { ArrowRight, BookOpen, Activity, PlayCircle, BarChart2, CheckCircle2, AlertCircle, FileText, Activity as ActivityIcon, Code, Network, XCircle, HelpCircle, Info } from 'lucide-react'

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
  t = t.replace(/20\/30\/Rest-Splits/g, '20/30/rest splits')
  t = t.replace(/Standard-Split/g, 'standard split')
  t = t.replace(/Horizont (\d+)/g, 'horizon $1')
  t = t.replace(/,(\d) %/g, '.$1%')
  return t
}

function colorForTone(tone: Tone) {
  if (tone === 'green') return 'text-tone-green'
  if (tone === 'amber') return 'text-tone-amber'
  if (tone === 'red') return 'text-tone-red'
  if (tone === 'violet') return 'text-tone-violet'
  return 'text-fg-4'
}

function iconForTone(tone: Tone) {
  if (tone === 'green') return <CheckCircle2 className="w-5 h-5 text-tone-green mt-0.5 shrink-0" />
  if (tone === 'amber') return <AlertCircle className="w-5 h-5 text-tone-amber mt-0.5 shrink-0" />
  if (tone === 'red') return <XCircle className="w-5 h-5 text-tone-red mt-0.5 shrink-0" />
  if (tone === 'violet') return <HelpCircle className="w-5 h-5 text-tone-violet mt-0.5 shrink-0" />
  return <Info className="w-5 h-5 text-fg-4 mt-0.5 shrink-0" />
}

// MSE/MAE are plain decimals, never percent; accuracy is a share shown as percent.
const showValue = (metric: string | undefined, x: any) => (metric && metric !== 'accuracy' ? (x == null ? '–' : Number(x).toFixed(3)) : toPercent(x))

function getDelta(measured: any, target: any, metric?: string) {
  if (measured === null || measured === undefined || target === null || target === undefined) return null
  const m = Number(measured)
  let t = Number(target)
  if (Number.isNaN(m) || Number.isNaN(t)) return null
  
  const plain = !!metric && metric !== 'accuracy'
  const mVal = plain ? m : (m <= 2.0 && m > 0) ? m * 100 : m
  const tVal = plain ? t : (t <= 2.0 && t > 0) ? t * 100 : t
  
  const diff = mVal - tVal
  if (plain) {
    if (Math.abs(diff) < 0.0005) return { text: '±0.000 vs. claimed', color: 'text-muted' }
    return { text: `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(3)} vs. claimed`, color: diff > 0 ? 'text-tone-green' : 'text-tone-red' }
  }
  // percentage points, same wording as the catalog
  if (Math.abs(diff) < 0.05) return { text: '±0.0 pp vs. claimed', color: 'text-muted' }
  if (diff > 0) return { text: `+${diff.toFixed(1)} pp vs. claimed`, color: 'text-tone-green' }
  return { text: `−${Math.abs(diff).toFixed(1)} pp vs. claimed`, color: 'text-tone-red' }
}

function getJuryStatement(claimed: string, measured: string, tone: Tone) {
  if (tone === 'green') return `The paper claims ${claimed}, and our independent sandbox measured ${measured}. The claim is fully verified and holds true.`
  if (tone === 'amber') return `The paper claims ${claimed}, but we measured ${measured}. The target was closely missed, indicating partial reproducibility.`
  if (tone === 'red') return `The paper claims ${claimed}, but we only measured ${measured}. The claim could not be verified.`
  return `The paper claims ${claimed}, but the verification is incomplete.`
}

// Catalog runs: always "with our setup", never a verdict on the paper.
function getCatalogStatement(claimed: string, measured: string, urteil: string, gesamt?: string) {
  if (gesamt === NOT_ASSESSABLE) return `The code did not implement the method, so the measurement (${measured}) says nothing about the reported ${claimed}.`
  if (urteil === 'reproduziert') return `Reproduced with our setup: measured ${measured} against the reported ${claimed}.`
  if (urteil === 'teilweise') return `Partially reproduced with our setup: measured ${measured} against the reported ${claimed}.`
  if (urteil === 'nicht reproduziert') return `Not reproduced with our setup: measured ${measured} against the reported ${claimed}. This is not a finding about the paper.`
  return `No valid measurement with our setup against the reported ${claimed}.`
}

function DashboardHeader({ count }: { count: number }) {
  return (
    <div className="mb-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-fg">Overview</h1>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-transparent border border-line text-fg-3 text-sm font-medium rounded-lg hover:bg-raised-2 transition-colors shadow-sm cursor-pointer">
            Filter runs
          </button>
          <a href="#discover" className="px-4 py-2 bg-inverse text-on-inverse border border-transparent text-sm font-medium rounded-lg hover:bg-inverse-hover hover:opacity-90 transition-all shadow-sm flex items-center gap-2 cursor-pointer no-underline">
            <PlayCircle className="w-4 h-4" />
            Discover papers
          </a>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-6 flex flex-col gap-2 rounded-xl border border-line bg-gradient-to-b from-raised to-panel shadow-sm">
          <div className="flex items-center gap-2 text-muted text-[10px] font-semibold uppercase tracking-widest">
            <Activity className="w-3.5 h-3.5 text-fg-4" />
            Total Runs
          </div>
          <span className="font-mono text-3xl font-semibold text-fg mt-1">{count}</span>
        </div>
        <div className="p-6 flex flex-col gap-2 rounded-xl border border-line bg-gradient-to-b from-raised to-panel shadow-sm">
          <div className="flex items-center gap-2 text-muted text-[10px] font-semibold uppercase tracking-widest">
            <CheckCircle2 className="w-3.5 h-3.5 text-tone-green" />
            Verified
          </div>
          <span className="font-mono text-3xl font-semibold text-fg mt-1">2</span>
        </div>
        <div className="p-6 flex flex-col gap-2 rounded-xl border border-line bg-gradient-to-b from-raised to-panel shadow-sm">
          <div className="flex items-center gap-2 text-muted text-[10px] font-semibold uppercase tracking-widest">
            <AlertCircle className="w-3.5 h-3.5 text-tone-amber" />
            Partial
          </div>
          <span className="font-mono text-3xl font-semibold text-fg mt-1">2</span>
        </div>
        <div className="p-6 flex flex-col gap-2 rounded-xl border border-line bg-gradient-to-b from-raised to-panel shadow-sm">
          <div className="flex items-center gap-2 text-muted text-[10px] font-semibold uppercase tracking-widest">
            <BarChart2 className="w-3.5 h-3.5 text-tone-blue" />
            Datasets
          </div>
          <span className="font-mono text-3xl font-semibold text-fg mt-1">3</span>
        </div>
      </div>
    </div>
  )
}

function Home({ runs }: { runs: RunRef[] }) {
  const [data, setData] = useState<Record<string, RunData>>({})
  useEffect(() => { runs.forEach((r) => loadRun(r).then((d) => setData((x) => ({ ...x, [r.id]: d })))) }, [runs])
  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-16">
      <DashboardHeader count={runs.length} />
      
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-[11px] font-semibold tracking-widest uppercase text-muted">Recent Evaluations</h2>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {runs.map((r, i) => {
          const d = data[r.id]
          const v = d?.dossier?.verdicts?.[0]
          const claim = d?.dossier?.claims?.[0]
          const res = v ? (r.catalog_id ? catalogLabel(v.urteil, r.gesamtaussage) : de2en(RESULT, v.urteil)) : null
          
          let delta = null
          if (v && claim) delta = getDelta(v.measured_mean, claim.target, claim.metric)
          
          return (
            <motion.a key={r.id} href={`#/run/${r.id}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className="flex flex-col p-6 rounded-xl border border-line bg-panel/50 no-underline transition-all duration-200 hover:border-line-strong hover:bg-raised/80 group cursor-pointer block">
              
              <div className="flex justify-between items-start mb-5">
                <div className="flex items-center gap-2 text-fg-4 text-[11px] font-medium font-mono uppercase tracking-wider">
                  <BookOpen className="w-3.5 h-3.5" />
                  {r.paper_id}
                </div>
                <div className="text-[11px] font-medium text-muted uppercase tracking-wider">{parseDate(r.timestamp)}</div>
              </div>
              
              <div className="flex flex-col gap-2.5 mb-5">
                {v ? (
                  <div className="flex flex-wrap items-center gap-3">
                    <Badge tone={res!.tone}>{res!.label}</Badge>
                    {r.catalog_id && isPreliminary(r.gesamtaussage) && <PreliminaryChip />}
                  </div>
                ) : <span className="text-[10px] text-muted uppercase tracking-widest font-semibold">Evaluating...</span>}
                <h3 className="text-lg font-semibold leading-snug text-fg group-hover:text-fg-max transition-colors">{r.title}</h3>
              </div>
              
              <p className="text-[13px] text-fg-4 line-clamp-2 leading-relaxed mb-6">
                {claim ? translateClaim(claim.text) : 'Evaluating target metrics...'}
              </p>
              
              <div className="mt-auto border-t border-line/50 pt-4 flex flex-col gap-2 w-full text-sm">
                {v ? (
                  <>
                    <div className="flex justify-between items-center w-full">
                      <span className="text-muted uppercase tracking-widest text-[10px] font-semibold">Claimed</span>
                      <span className="font-mono text-fg-4">{showValue(claim?.metric, claim?.target)}</span>
                    </div>
                    <div className="flex justify-between items-start w-full pt-1">
                      <span className="text-muted uppercase tracking-widest text-[10px] font-semibold mt-1">Measured</span>
                      <div className="flex flex-col items-end gap-1">
                        <span className={`font-mono font-bold text-lg leading-none ${colorForTone(res!.tone)}`}>{showValue(claim?.metric, v.measured_mean)}</span>
                        {delta && <span className={`font-mono text-[10px] ${delta.color}`}>{delta.text}</span>}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="h-10 flex items-center">
                    <span className="text-xs text-subtle font-mono">Running sandbox...</span>
                  </div>
                )}
              </div>
            </motion.a>
          )
        })}
      </div>
    </main>
  )
}

function RunView({ ref_, initialTab }: { ref_: RunRef; initialTab?: string }) {
  const [run, setRun] = useState<RunData | null>(null)
  const [selected, setSelected] = useState<Json | null>(null)
  const [activeTab, setActiveTab] = useState<'evidence' | 'tree' | 'log' | 'files'>(initialTab === 'log' ? 'log' : 'evidence')
  useEffect(() => { loadRun(ref_).then(setRun) }, [ref_])
  
  if (!run) return <main className="mx-auto max-w-6xl px-4 md:px-8 py-10 text-muted">Loading run data…</main>
  const d = run.dossier
  const claim: Json = d.claims?.[0]
  const v: Json = d.verdicts?.[0]
  if (!claim) return <main className="mx-auto max-w-6xl px-4 md:px-8 py-10 text-muted">Empty run.</main>
  
  const isCatalog = !!ref_.catalog_id   // discovery candidate run through the loop
  const res = v ? (isCatalog ? catalogLabel(v.urteil, ref_.gesamtaussage) : de2en(RESULT, v.urteil)) : { label: 'NO VERDICT', tone: 'gray' as const }
  const attempts: Json[] = (d.attempts ?? []).filter((a: Json) => a.claim_id === claim.id)
  const paperUrl = claim.target_evidence?.url
  
  let delta = null
  if (v && claim) delta = getDelta(v.measured_mean, claim.target, claim.metric)
  const resColorClass = colorForTone(res.tone)

  const tabs = [
    { id: 'evidence', label: 'Data & Evidence', icon: <FileText className="w-4 h-4" /> },
    { id: 'log', label: 'Execution Log', icon: <ActivityIcon className="w-4 h-4" /> },
    { id: 'tree', label: 'Branch Tree', icon: <Network className="w-4 h-4" /> },
    { id: 'files', label: 'Raw Files', icon: <Code className="w-4 h-4" /> },
  ] as const

  return (
    <main className="mx-auto max-w-6xl space-y-8 px-4 md:px-8 pb-20">
      <div className="pt-2">
        <a href="#" className="inline-flex items-center gap-1.5 text-muted text-xs font-medium uppercase tracking-wider hover:text-fg-2 transition-colors">
          <ArrowRight className="w-3.5 h-3.5 rotate-180" /> Back to Overview
        </a>
      </div>
      
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="pb-8">
        <div className="flex items-center gap-3 text-muted text-xs mb-3 font-mono">
          <span>{run.ref.paper_id}</span>
          <span>•</span>
          <span>{parseDate(run.ref.timestamp)}</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-fg leading-tight">{run.ref.title}</h1>
        <p className="mt-4 text-lg text-fg-4 max-w-3xl leading-relaxed">{translateClaim(claim.text)}</p>
        
        {paperUrl && (
          <a className="inline-flex items-center gap-2 mt-5 text-sm font-medium text-tone-blue hover:text-tone-blue-hover transition-colors" href={paperUrl} target="_blank" rel="noreferrer">
            <BookOpen className="w-4 h-4" />
            View Source Paper
          </a>
        )}
      </motion.div>

      {isCatalog && <CatalogCallout run={run} claimId={claim.id} gesamt={ref_.gesamtaussage} />}

      {/* Split Layout: Verdict & Chart */}
      <div className="grid gap-6 lg:grid-cols-3 items-stretch">
        <Section title="Evaluation Verdict" className="lg:col-span-1 border-t-2" style={{ borderTopColor: TONE_VAR[res.tone] }}>
          <div className="mb-8">
            <h3 className={`text-4xl font-bold tracking-tight mt-2 ${resColorClass}`}>{res.label}</h3>
          </div>
          
          <div className="space-y-4 mb-8">
            <div className="flex justify-between items-center w-full">
              <span className="text-muted uppercase tracking-widest text-[10px] font-semibold">Claimed</span>
              <span className="font-mono text-fg-3 text-lg">{showValue(claim.metric, claim.target)}</span>
            </div>
            <div className="flex justify-between items-start w-full pt-4 border-t border-line/50">
              <span className="text-muted uppercase tracking-widest text-[10px] font-semibold mt-1">Measured</span>
              <div className="flex flex-col items-end gap-1.5">
                <span className={`font-mono font-bold text-3xl leading-none ${resColorClass}`}>{showValue(claim.metric, v?.measured_mean)}</span>
                {delta && <span className={`font-mono text-[11px] mt-1 ${delta.color}`}>{delta.text}</span>}
              </div>
            </div>
          </div>
          
          <div className="flex items-start gap-3 mt-8">
            {iconForTone(res.tone)}
            <div>
              <p className="text-[13px] text-fg-3 leading-relaxed font-medium">
                {isCatalog && v
                  ? getCatalogStatement(showValue(claim.metric, claim.target), showValue(claim.metric, v.measured_mean), v.urteil, ref_.gesamtaussage)
                  : getJuryStatement(showValue(claim.metric, claim.target), showValue(claim.metric, v?.measured_mean), res.tone)}
              </p>
              <p className="text-[11px] text-muted mt-2 font-mono">Margin of Error: ±{claim.metric === 'accuracy' ? toPercent(claim.tolerance) : Number(claim.tolerance).toFixed(3)}</p>
            </div>
          </div>
        </Section>
        
        <Section title="Measurements" hint={`${attempts.length} valid runs`} className="lg:col-span-2 flex flex-col">
          <AttemptChart dossier={d} claimId={claim.id} />
        </Section>
      </div>

      {/* Tabs Layout */}
      <div className="mt-12">
        <div className="flex gap-2 border-b border-line/50 pb-px mb-6">
          {tabs.map((t) => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} 
              aria-current={activeTab === t.id ? 'page' : undefined}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-colors border-b-2 -mb-px cursor-pointer ${activeTab === t.id ? 'border-fg text-fg' : 'border-transparent text-muted hover:text-fg-3 hover:border-line-strong/50'}`}>
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
        
        <AnimatePresence mode="wait">
          <motion.div key={activeTab} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: 0.15 }}>
            {activeTab === 'evidence' && <EvidenceSplit claimId={claim.id} dossier={d} events={run.events} selected={selected} />}
            {activeTab === 'log' && <Replay events={run.events} />}
            {activeTab === 'tree' && <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />}
            {activeTab === 'files' && (
              <div className="space-y-6">
                <div className="p-4 bg-tone-amber-solid/10 border border-tone-amber-solid/20 text-tone-amber rounded-lg text-sm flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <p>These raw files are primarily used by developers to debug the agent's behavior. They contain internal Markdown structure.</p>
                </div>
                <div className="flex gap-2">
                  <button className="px-4 py-2 text-xs font-semibold uppercase tracking-wider rounded-lg bg-raised border border-line text-fg-3 hover:bg-raised-2">Agent Code</button>
                  <button className="px-4 py-2 text-xs font-semibold uppercase tracking-wider rounded-lg bg-raised border border-line text-fg-3 hover:bg-raised-2">Execution Plan</button>
                </div>
                <pre className="font-mono h-96 overflow-auto whitespace-pre-wrap rounded-xl p-5 bg-surface border border-line/50 text-fg-4 text-[12px] leading-relaxed">
                  {run.code || 'No code snapshot available.'}
                </pre>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  )
}

export default function App() {
  const hash = useHash()
  const [runs, setRuns] = useState<RunRef[] | null>(null)
  useEffect(() => {
    const load = () => loadIndex().then(setRuns)
    load()
    window.addEventListener(DATA_CHANGED, load)   // a live run finished: index.json has a new entry
    return () => window.removeEventListener(DATA_CHANGED, load)
  }, [])
  const m = hash.match(/^\/run\/([^?]+)(?:\?tab=(\w+))?$/)
  const ref = m && runs?.find((r) => r.id === decodeURIComponent(m[1]))
  const live = hash.match(/^\/live\/([^?]+)$/)
  
  return (
    <Layout currentHash={hash}>
      <div className="py-6 min-h-screen bg-bg">
        {hash === 'discover' ? <Discovery /> : hash === 'catalog' ? <Catalog /> : live ? <LiveRun jobId={decodeURIComponent(live[1])} key={live[1]} /> : runs === null ? (
          <div className="mx-auto max-w-6xl px-4 md:px-8 py-20 flex flex-col items-center justify-center gap-4">
            <Activity className="w-8 h-8 text-subtle animate-pulse" />
            <span className="text-muted text-sm font-medium uppercase tracking-wider">Loading Evaluation Data...</span>
          </div>
        ) : ref ? <RunView ref_={ref} initialTab={m?.[2]} key={ref.id} /> : <Home runs={runs} />}
      </div>
    </Layout>
  )
}

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Layout } from './components/Layout'
import { Discovery } from './components/Discovery'
import { Catalog } from './components/Catalog'
import { LiveRun } from './components/LiveRun'
import { Landing } from './components/Landing'
import { DATA_CHANGED } from './live'
import { PreliminaryChip } from './components/CatalogBits'
import { NextStep, PageIntro } from './components/page'
import { RunView } from './components/RunView'
import { Benchmark } from './components/Benchmark'
import { colorForTone, getDelta, parseDate, showValue } from './format'
import { Badge } from './components/ui'
import { loadIndex, loadRun, type RunData, type RunRef } from './data'
import { RESULT, catalogLabel, de2en, isPreliminary, translateClaim, type Tone } from './labels'
import { BookOpen, Activity } from 'lucide-react'

const useHash = () => {
  const [h, setH] = useState(location.hash.slice(1))
  useEffect(() => {
    const f = () => setH(location.hash.slice(1))
    addEventListener('hashchange', f)
    return () => removeEventListener('hashchange', f)
  }, [])
  return h
}

// Counters come from the run index and the loaded dossiers; nothing is typed in by hand.
function Counters({ items }: { items: { label: string; value: number; tone?: Tone }[] }) {
  return (
    <dl className="m-0 grid grid-cols-2 gap-0 border-y border-line md:grid-cols-4">
      {items.map((it, i) => (
        <div key={it.label} className={`py-6 ${i % 2 ? 'border-l border-line pl-6' : ''} ${i > 1 ? 'border-t border-line md:border-t-0' : ''} ${i > 0 ? 'md:border-l md:border-line md:pl-8' : ''}`}>
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">{it.label}</dt>
          <dd className={`m-0 mt-2 font-mono text-4xl font-light tracking-tight ${it.tone ? colorForTone(it.tone) : 'text-fg'}`}>{it.value}</dd>
        </div>
      ))}
    </dl>
  )
}

function Home({ runs }: { runs: RunRef[] }) {
  const [data, setData] = useState<Record<string, RunData>>({})
  useEffect(() => { runs.forEach((r) => loadRun(r).then((d) => setData((x) => ({ ...x, [r.id]: d })))) }, [runs])
  const urteil = (r: RunRef) => data[r.id]?.dossier?.verdicts?.[0]?.urteil
  const reproduced = runs.filter((r) => urteil(r) === 'reproduziert')
  const partial = runs.filter((r) => urteil(r) === 'teilweise').length
  const papers = new Set(runs.map((r) => r.paper_id)).size
  const example = reproduced[0] ?? runs[0]
  // several runs of one paper: say which one a card is ("Run 2 of 3 for this paper"), oldest first
  const sameCount = (r: RunRef) => runs.filter((x) => x.paper_id === r.paper_id).length
  const runNo = (r: RunRef) => runs.filter((x) => x.paper_id === r.paper_id).sort((a, b) => a.timestamp.localeCompare(b.timestamp)).findIndex((x) => x.id === r.id) + 1
  return (
    <main className="mx-auto max-w-6xl px-4 pb-16 md:px-8">
      <PageIntro kicker="Runs" title="Finished checks"
        what="Each card is one finished check. A paper reported a number, Nullius rebuilt the experiment and measured it again. Reproduced means our number landed within the allowed tolerance of the claim."
        look="Compare Claimed and Measured on any card. Open a card to see the evidence behind its verdict." />
      <Counters items={[
        { label: 'Checks', value: runs.length }, { label: 'Reproduced', value: reproduced.length, tone: 'green' },
        { label: 'Partially', value: partial, tone: 'amber' }, { label: 'Papers', value: papers },
      ]} />

      <div className="mb-6 mt-14 flex items-baseline justify-between gap-4">
        <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">All checks</h2>
        <span className="text-[12px] text-muted">Measured with our setup. Method fidelity not checked.</span>
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
            <motion.a key={r.id} href={`#/run/${r.id}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
              className="group flex cursor-pointer flex-col rounded-xl border border-line bg-surface p-6 no-underline transition-colors duration-200 hover:border-line-strong">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                {v ? (
                  <>
                    <Badge tone={res!.tone}>{res!.label}</Badge>
                    {r.catalog_id && isPreliminary(r.gesamtaussage) && <PreliminaryChip />}
                  </>
                ) : <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">Evaluating...</span>}
              </div>
              <h3 className="font-grotesk m-0 text-xl font-semibold leading-snug tracking-[-0.02em] text-fg">{r.title}</h3>
              <p className="mb-6 mt-3 line-clamp-2 text-[14px] leading-relaxed text-fg-4">{claim ? translateClaim(claim.text) : 'Evaluating target metrics...'}</p>
              <div className="mt-auto flex w-full flex-col gap-2 border-t border-line pt-4 text-sm">
                {v ? (
                  <>
                    <div className="flex w-full items-center justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-widest text-muted">Claimed</span>
                      <span className="font-mono text-fg-4">{showValue(claim?.metric, claim?.target)}</span>
                    </div>
                    <div className="flex w-full items-start justify-between pt-1">
                      <span className="mt-1 text-[10px] font-semibold uppercase tracking-widest text-muted">Measured</span>
                      <div className="flex flex-col items-end gap-1">
                        <span className={`font-mono text-lg font-bold leading-none ${colorForTone(res!.tone)}`}>{showValue(claim?.metric, v.measured_mean)}</span>
                        {delta && <span className={`font-mono text-[10px] ${delta.color}`}>{delta.text}</span>}
                      </div>
                    </div>
                  </>
                ) : <span className="font-mono text-xs text-subtle">Running sandbox...</span>}
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted">
                  <span className="inline-flex items-center gap-1.5"><BookOpen className="h-3 w-3" />{sameCount(r) > 1 ? `Run ${runNo(r)} of ${sameCount(r)} for this paper` : 'One run for this paper'}</span>
                  <span>{parseDate(r.timestamp)}</span>
                </div>
              </div>
            </motion.a>
          )
        })}
      </div>

      <NextStep title={example ? 'See how one check ends' : 'Find a paper to check'}
        text={example ? 'Open a finished check: the verdict in one sentence, the steps the agent took, and why the number can be trusted.' : 'No checks yet. Discover papers that report results on a benchmark we can rebuild.'}
        label={example ? 'Open an example' : 'Discover papers'} href={example ? `#/run/${example.id}` : '#discover'} />
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
  const m = hash.match(/^\/run\/([^?]+)(?:\?(.*))?$/)
  const query = new URLSearchParams(m?.[2] ?? '')
  const ref = m && runs?.find((r) => r.id === decodeURIComponent(m[1]))
  const live = hash.match(/^\/live\/([^?]+)$/)
  
  if (hash === '' || hash === '/') return <Landing />   // full-screen start page, no sidebar

  return (
    <Layout currentHash={hash}>
      <div className="py-6 min-h-screen bg-bg">
        {hash === 'discover' ? <Discovery /> : hash === 'catalog' ? <Catalog /> : hash === 'benchmark' ? <Benchmark /> : live ? <LiveRun jobId={decodeURIComponent(live[1])} key={live[1]} /> : runs === null ? (
          <div className="mx-auto max-w-6xl px-4 md:px-8 py-20 flex flex-col items-center justify-center gap-4">
            <Activity className="w-8 h-8 text-subtle animate-pulse" />
            <span className="text-muted text-sm font-medium uppercase tracking-wider">Loading Evaluation Data...</span>
          </div>
        ) : ref ? <RunView ref_={ref} initialTab={query.get('tab') ?? undefined} playTimelapse={query.get('play') === '1'} key={ref.id} /> : <Home runs={runs} />}
      </div>
    </Layout>
  )
}

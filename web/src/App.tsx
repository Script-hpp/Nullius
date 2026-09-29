import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { AttemptChart } from './components/Gauge'
import { BranchTree } from './components/BranchTree'
import { EvidenceSplit } from './components/Evidence'
import { Replay } from './components/Replay'
import { Badge, Section, Stat, fmt } from './components/ui'
import { loadIndex, loadRun, type Json, type RunData, type RunRef } from './data'
import { METHOD, OVERALL, RESULT, de2en } from './labels'

const useHash = () => {
  const [h, setH] = useState(location.hash.slice(1))
  useEffect(() => {
    const f = () => setH(location.hash.slice(1))
    addEventListener('hashchange', f)
    return () => removeEventListener('hashchange', f)
  }, [])
  return h
}

function Header() {
  return (
    <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-6">
      <a href="#" className="flex items-baseline gap-3 no-underline" style={{ color: 'var(--text)' }}>
        <span className="text-lg font-semibold">Paper reproduction agent</span>
        <span className="muted hidden text-sm sm:inline">— the paper is the test suite</span>
      </a>
      <span className="muted text-xs">evidence view · replay of real runs</span>
    </header>
  )
}

function Home({ runs }: { runs: RunRef[] }) {
  const [data, setData] = useState<Record<string, RunData>>({})
  useEffect(() => { runs.forEach((r) => loadRun(r).then((d) => setData((x) => ({ ...x, [r.id]: d })))) }, [runs])
  return (
    <main className="mx-auto max-w-6xl px-5 pb-16">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-8 max-w-3xl">
        <h1 className="text-3xl font-semibold leading-tight">Other agents fix code until tests turn green.<br />Here, the paper is the test suite.</h1>
        <p className="muted mt-3">
          Each card is a real run: the agent extracts a claim from an arXiv paper, Tavily supplies target values and setup details, the sandbox re-runs the experiment in branches,
          and an independent harness scores it. Open a run to replay it step by step.
        </p>
      </motion.div>
      <div className="grid gap-4 md:grid-cols-2">
        {runs.map((r, i) => {
          const d = data[r.id]
          const v = d?.dossier?.verdicts?.[0]
          const claim = d?.dossier?.claims?.[0]
          const res = v ? de2en(RESULT, v.urteil) : null
          return (
            <motion.a key={r.id} href={`#/run/${r.id}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
              className="panel block p-5 no-underline transition hover:-translate-y-0.5" style={{ color: 'var(--text)' }}>
              <div className="muted text-xs">{r.paper_id} · {r.timestamp}</div>
              <div className="mt-1 text-lg font-medium">{r.title}</div>
              {claim && <p className="muted mt-1 text-sm">{claim.text}</p>}
              {v && (
                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <Badge tone={res!.tone}>{res!.label}</Badge>
                  <Badge tone={de2en(METHOD, v.methode).tone}>method: {de2en(METHOD, v.methode).label}</Badge>
                  <span className="mono muted ml-auto text-xs">{fmt(v.measured_mean, 3)} vs {claim?.target}</span>
                </div>
              )}
              {r.note && <p className="muted mt-3 text-xs">{r.note}</p>}
            </motion.a>
          )
        })}
      </div>
      {runs.length === 0 && <p className="muted">No runs exported yet. Run <code>scripts/export_site_data.py</code>.</p>}
    </main>
  )
}

function RunView({ ref_ }: { ref_: RunRef }) {
  const [run, setRun] = useState<RunData | null>(null)
  const [selected, setSelected] = useState<Json | null>(null)
  const [tab, setTab] = useState<'code' | 'zeugnis' | 'plan'>('zeugnis')
  useEffect(() => { loadRun(ref_).then(setRun) }, [ref_])
  if (!run) return <main className="mx-auto max-w-6xl px-5 py-10 muted">Loading…</main>
  const d = run.dossier
  const claim: Json = d.claims?.[0]
  const v: Json = d.verdicts?.[0]
  if (!claim) return <main className="mx-auto max-w-6xl px-5 py-10 muted">Empty run.</main>
  const res = v ? de2en(RESULT, v.urteil) : { label: 'no verdict', tone: 'gray' as const }
  const meth = v ? de2en(METHOD, v.methode) : { label: '–', tone: 'gray' as const }
  const attempts: Json[] = (d.attempts ?? []).filter((a: Json) => a.claim_id === claim.id)
  const end = run.events.find((e) => e.type === 'run_end')
  const cost = end?.cost_net_usd ?? attempts[attempts.length - 1]?.cost_usd
  const paperUrl = claim.target_evidence?.url
  const tabs = { zeugnis: run.zeugnis, code: run.code, plan: run.plan }
  return (
    <main className="mx-auto max-w-6xl space-y-5 px-5 pb-20">
      <a href="#" className="muted text-sm no-underline">← all runs</a>
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="muted text-xs">{run.ref.paper_id} · {run.ref.timestamp} · mode: {d.mode === 'nachbau' ? 'rebuild (own code)' : d.mode}</div>
        <h1 className="mt-1 text-2xl font-semibold">{run.ref.title}</h1>
        <p className="mt-2 max-w-3xl">{claim.text}</p>
        {paperUrl && <a className="text-xs" style={{ color: 'var(--blue)' }} href={paperUrl} target="_blank" rel="noreferrer">{paperUrl}</a>}
      </motion.div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Section title="Verdict" hint="two axes" className="lg:col-span-1">
          <div className="space-y-3">
            <div><div className="muted mb-1 text-xs">1 · Result (mean vs tolerance)</div><Badge tone={res.tone}>{res.label}</Badge></div>
            <div><div className="muted mb-1 text-xs">2 · Method fidelity</div><Badge tone={meth.tone}>{meth.label}</Badge></div>
            <div><div className="muted mb-1 text-xs">Overall statement</div><span className="text-sm font-medium">{OVERALL[v?.gesamt ?? ''] ?? (v?.methode === 'nicht geprueft' ? 'provisional' : '–')}</span></div>
            <p className="muted text-xs">A hit on the number alone proves nothing about the method: a closed-form regression can hit a target value, so the second axis exists.</p>
          </div>
          <div className="mt-5 grid grid-cols-3 gap-3 border-t pt-4" style={{ borderColor: 'var(--line)' }}>
            <Stat label="measured" value={fmt(v?.measured_mean, 4)} sub={`± ${fmt(v?.measured_std, 4)} std`} />
            <Stat label="target" value={claim.target} sub={`± ${claim.tolerance}`} />
            <Stat label="cost" value={cost != null ? `$${Number(cost).toFixed(3)}` : '–'} sub="net" />
          </div>
        </Section>
        <Section title="Measurements" hint={`${attempts.length} runs, one bar each`} className="lg:col-span-2">
          <AttemptChart dossier={d} claimId={claim.id} />
        </Section>
      </div>

      <Section title="Evidence" hint="left: what Tavily found · right: what the sandbox measured">
        <EvidenceSplit claimId={claim.id} dossier={d} events={run.events} selected={selected} />
      </Section>

      <Section title="Branch tree" hint="click a run for details · violet edges = repair after an error">
        <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />
      </Section>

      <Section title="Replay" hint="the run as it happened">
        <Replay events={run.events} />
      </Section>

      <Section title="Files">
        <div className="mb-3 flex gap-2">
          {(['zeugnis', 'code', 'plan'] as const).map((t) => tabs[t] != null && (
            <button key={t} onClick={() => setTab(t)} className="rounded-lg px-3 py-1 text-sm" style={{ background: tab === t ? 'var(--blue)' : 'var(--panel-2)', color: tab === t ? '#04101d' : 'var(--text)' }}>
              {t === 'zeugnis' ? 'certificate' : t === 'code' ? 'agent code' : 'plan'}
            </button>
          ))}
        </div>
        <pre className="mono max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-lg p-4 text-xs panel-2">{tabs[tab] ?? 'not available'}</pre>
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
    <>
      <Header />
      {runs === null ? <main className="mx-auto max-w-6xl px-5 muted">Loading…</main> : ref ? <RunView ref_={ref} key={ref.id} /> : <Home runs={runs} />}
    </>
  )
}

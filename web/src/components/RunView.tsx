import { useCallback, useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, Code, Play, FileText, Network, Workflow, Activity as ActivityIcon, ArrowRight } from 'lucide-react'
import { BranchTree } from './BranchTree'
import { CatalogCallout } from './CatalogBits'
import { EvidenceSplit } from './Evidence'
import { NextStep, PageIntro } from './page'
import { RepairStory } from './RepairStory'
import { ResultBand } from './ResultChart'
import { ReplayPlayer } from './run/ReplayPlayer'
import { StepGraph } from './run/StepGraph'
import { StepStrip, buildSteps, type StepId } from './run/StepStrip'
import { PaperHead } from './PaperHead'
import { Film } from './film/Film'
import { DropLast } from './DropLast'
import { ReadmeBadge } from './ReadmeBadge'
import { CommunityCard } from './CommunityCard'
import { Trust } from './Trust'
import { WhatIf } from './WhatIf'
import { toPercent } from './ui'
import { useHead } from '../useHead'
import { headFromSummary } from '../paperhead'
import { colorForTone, getCatalogStatement, getDelta, iconForTone, parseDate, resultSentence, showValue } from '../format'
import { getText, loadRun, runDir, type Json, type RunData, type RunRef } from '../data'
import { OVERALL, RESULT, TONE_VAR, catalogLabel, de2en, translateClaim } from '../labels'

type Tab = 'graph' | 'evidence' | 'tree' | 'log' | 'files'

const scrollTo = (id: string) => {
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  document.getElementById(id)?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
}

// The certificate of one check: verdict in one sentence, claimed against measured, why to trust it, the nine steps, then the details.
export function RunView({ ref_, initialTab, playTimelapse = false }: { ref_: RunRef; initialTab?: string; playTimelapse?: boolean }) {
  const [run, setRun] = useState<RunData | null>(null)
  const [selected, setSelected] = useState<Json | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>(initialTab === 'log' || initialTab === 'evidence' || initialTab === 'tree' || initialTab === 'files' ? initialTab : 'graph')
  const [file, setFile] = useState<'code' | 'plan'>('code')
  const [timelapse, setTimelapse] = useState(playTimelapse)   // the 30-second panel, opened by the button or by ?play=1
  const stored = useHead(ref_.paper_id)
  useEffect(() => { loadRun(ref_).then(setRun) }, [ref_])
  // code versions of the replay: code_<hash>.py from the export, older exports only have the final code.py
  const finalHash = useMemo(() => {
    const att: Json[] = (run?.dossier?.attempts ?? []).filter((x: Json) => x.measured != null)
    return att.length ? att[att.length - 1].code_hash : null
  }, [run])
  const codeLoader = useCallback((hash: string) => getText(`${runDir(ref_)}code_${hash}.py`).then((t) => t ?? (hash === finalHash ? run?.code ?? null : null)), [ref_, finalHash, run])

  if (!run) return <main className="mx-auto max-w-6xl px-4 py-10 text-muted md:px-8">Loading run data…</main>
  const d = run.dossier
  const head = headFromSummary(d.summary) ?? stored   // what this run wrote itself, else the stored head of the paper
  const claim: Json = d.claims?.[0]
  const v: Json = d.verdicts?.[0]
  if (!claim) return <main className="mx-auto max-w-6xl px-4 py-10 text-muted md:px-8">Empty run.</main>

  const isCatalog = !!ref_.catalog_id   // discovery candidate run through the loop
  const res = v ? (isCatalog ? catalogLabel(v.urteil, ref_.gesamtaussage) : de2en(RESULT, v.urteil)) : { label: 'NO VERDICT', tone: 'gray' as const }
  const paperUrl = claim.target_evidence?.url
  const delta = v ? getDelta(v.measured_mean, claim.target, claim.metric) : null
  const resColor = colorForTone(res.tone)
  const fidelityUnchecked = !v?.methode || v.methode === 'nicht geprueft'
  const fmt = (x: number) => showValue(claim.metric, x)
  const sentence = resultSentence(claim, v, ref_.gesamtaussage)   // one string for the verdict section and the time-lapse
  const steps = buildSteps(run.events, d, { running: false, hasPlan: !!run.plan, metric: claim.metric })

  const tabs = [
    { id: 'graph', label: 'Step graph', icon: <Workflow className="h-4 w-4" /> },
    { id: 'evidence', label: 'Data & evidence', icon: <FileText className="h-4 w-4" /> },
    { id: 'log', label: 'Execution log', icon: <ActivityIcon className="h-4 w-4" /> },
    { id: 'tree', label: 'Branch tree', icon: <Network className="h-4 w-4" /> },
    { id: 'files', label: 'Plan & code', icon: <Code className="h-4 w-4" /> },
  ] as const

  const onStep = (id: StepId) => {
    const go = (tab: Tab, f?: 'code' | 'plan') => { setActiveTab(tab); if (f) setFile(f); window.setTimeout(() => scrollTo('sec-details'), 30) }
    if (id === 'paper' || id === 'verdict') scrollTo('sec-verdict')
    else if (id === 'claim' || id === 'sources') go('evidence')
    else if (id === 'plan') go('files', 'plan')
    else if (id === 'code') go('files', 'code')
    else if (id === 'sandbox') go('tree')
    else if (id === 'check') go('log')
    else if (id === 'community') scrollTo('sec-community')
  }

  return (
    <main className="mx-auto max-w-6xl space-y-16 px-4 pb-20 md:px-8">
      {timelapse && run.events.length > 0 && (
        <Film events={run.events} dossier={d} claim={claim} title={run.ref.title} tone={res.tone} resultLabel={res.label} sentence={sentence} fidelityUnchecked={fidelityUnchecked}
          overall={OVERALL[ref_.gesamtaussage ?? ''] ?? null} fmt={fmt} fmtSpread={(x) => (claim.metric === 'accuracy' ? (x * 100).toFixed(1) : x.toFixed(3))}
          plan={run.plan} finalCode={run.code} codeLoader={codeLoader} autoplay />
      )}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-3 pt-6">
          <a href="#overview" className="inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted no-underline transition-colors hover:text-fg-2">
            <ArrowRight className="h-3.5 w-3.5 rotate-180" /> Back to runs
          </a>
          {run.events.length > 0 && !timelapse && (
            <button type="button" onClick={() => setTimelapse(true)} className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-line-strong bg-transparent px-4 py-2 text-[13px] font-medium text-fg hover:bg-raised/60">
              <Play className="h-3.5 w-3.5" />Play this check in 30 seconds
            </button>
          )}
        </div>
        <PageIntro kicker={`${run.ref.paper_id} · ${parseDate(run.ref.timestamp)}`} title={run.ref.title} head={<PaperHead head={head} />}
          what="This is the certificate of one check. The paper claimed a number; our agent rebuilt the experiment and an independent program measured it."
          look="Read the verdict sentence first, then compare claimed and measured. The nine steps below show how we got there, and why the number can be trusted." />
        <p className="m-0 max-w-3xl text-lg leading-relaxed text-fg-4">Claim: {translateClaim(claim.text)}</p>
        {paperUrl && (
          <a className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-tone-blue no-underline transition-colors hover:text-tone-blue-hover" href={paperUrl} target="_blank" rel="noreferrer">
            <BookOpen className="h-4 w-4" />View the source paper
          </a>
        )}
      </div>

      {isCatalog && <CatalogCallout run={run} claimId={claim.id} gesamt={ref_.gesamtaussage} />}

      <section id="sec-verdict" className="scroll-mt-4 border-t-2 pt-10" style={{ borderTopColor: TONE_VAR[res.tone] }} aria-label="The verdict">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">The verdict in one sentence</div>
        <p className="font-grotesk m-0 mt-4 max-w-4xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">{sentence}</p>
        <div className="mt-10 grid items-start gap-10 lg:grid-cols-3">
          <div className="min-w-0">
            <h2 className={`font-grotesk m-0 text-4xl font-semibold tracking-[-0.04em] ${resColor}`}>{res.label}</h2>
            <dl className="m-0 mt-8 space-y-4">
              <div className="flex items-center justify-between border-b border-line pb-4">
                <dt className="text-[10px] font-semibold uppercase tracking-widest text-muted">Claimed</dt>
                <dd className="m-0 font-mono text-lg text-fg-3">{fmt(claim.target)}</dd>
              </div>
              <div className="flex items-start justify-between">
                <dt className="mt-1 text-[10px] font-semibold uppercase tracking-widest text-muted">Measured</dt>
                <dd className="m-0 flex flex-col items-end gap-1.5">
                  <span className={`font-mono text-3xl font-bold leading-none ${resColor}`}>{fmt(v?.measured_mean)}</span>
                  {delta && <span className={`font-mono text-[11px] ${delta.color}`}>{delta.text}</span>}
                </dd>
              </div>
            </dl>
            <div className="mt-6 flex items-start gap-3">
              {iconForTone(res.tone)}
              <div>
                <p className="m-0 text-[13px] font-medium leading-relaxed text-fg-3">
                  {v ? getCatalogStatement(fmt(claim.target), fmt(v.measured_mean), v.urteil, ref_.gesamtaussage) : `No verdict yet for the claimed ${fmt(claim.target)}.`}
                </p>
                <p className="mb-0 mt-2 font-mono text-[11px] text-muted">Allowed tolerance: ±{claim.metric === 'accuracy' ? toPercent(claim.tolerance) : Number(claim.tolerance).toFixed(3)}</p>
              </div>
            </div>
          </div>
          <div className="min-w-0 lg:col-span-2">
            <ResultBand dossier={d} claimId={claim.id} tone={res.tone} fmt={fmt} fmtSpread={(x) => (claim.metric === 'accuracy' ? (x * 100).toFixed(1) : x.toFixed(3))} />
          </div>
        </div>
      </section>

      <WhatIf dossier={d} claim={claim} verdict={v} fmt={fmt} fmtSpread={(x) => (claim.metric === 'accuracy' ? (x * 100).toFixed(1) : x.toFixed(3))} />

      <CommunityCard dossier={d} claimId={claim.id} />

      <Trust fidelityUnchecked={fidelityUnchecked} />

      <DropLast dossier={d} claimId={claim.id} />

      <StepStrip steps={steps} onStep={onStep} />

      <RepairStory events={run.events} claimId={claim.id} codeLoader={codeLoader} />

      <ReadmeBadge ref_={ref_} />

      <section id="sec-details" className="scroll-mt-4 min-w-0" aria-label="Details">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Details</div>
        <p className="mb-0 mt-3 max-w-3xl text-[15px] leading-relaxed text-fg-4">Everything behind the verdict: the step graph, the data and the quoted evidence, the log of the run, the sandbox branches, and the exact plan and code.</p>
        <div className="mb-6 mt-8 flex gap-2 overflow-x-auto border-b border-line pb-px">
          {tabs.map((t) => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} aria-current={activeTab === t.id ? 'page' : undefined}
              className={`-mb-px flex shrink-0 cursor-pointer items-center gap-2 border-0 border-b-2 bg-transparent px-5 py-3 text-sm font-medium transition-colors ${activeTab === t.id ? 'border-fg text-fg' : 'border-transparent text-muted hover:border-line-strong/50 hover:text-fg-3'}`}>
              {t.icon}{t.label}
            </button>
          ))}
        </div>
        <AnimatePresence mode="wait">
          <motion.div key={activeTab} initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: 0.15 }}>
            {activeTab === 'graph' && <StepGraph events={run.events} running={false} codeLoader={codeLoader} planFull={run.plan} metricHint={claim.metric} />}
            {activeTab === 'evidence' && <EvidenceSplit claimId={claim.id} dossier={d} events={run.events} selected={selected} />}
            {activeTab === 'log' && <ReplayPlayer events={run.events} codeLoader={codeLoader} planFull={run.plan} />}
            {activeTab === 'tree' && <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />}
            {activeTab === 'files' && (
              <div className="space-y-5">
                <p className="m-0 text-[13px] leading-relaxed text-muted">The plan the agent followed and the code it wrote for this run, exactly as stored.</p>
                <div className="flex gap-2">
                  {(['code', 'plan'] as const).map((f) => (
                    <button key={f} onClick={() => setFile(f)} aria-pressed={file === f}
                      className={`cursor-pointer rounded-lg border px-4 py-2 text-xs font-semibold uppercase tracking-wider ${file === f ? 'border-line-stronger bg-raised text-fg' : 'border-line text-muted hover:bg-raised-2'}`}>
                      {f === 'code' ? 'Agent code' : 'Plan'}
                    </button>
                  ))}
                </div>
                <pre className="m-0 h-96 overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-surface p-5 font-mono text-[12px] leading-relaxed text-fg-4">
                  {file === 'code' ? run.code || 'No code snapshot available.' : run.plan || 'No plan recorded for this run.'}
                </pre>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </section>

      <NextStep title="See how often Nullius is right" text="One check is an example. The benchmark runs the whole procedure on papers with known results and counts how often it agrees, with and without Tavily."
        label="Open the benchmark" href="#benchmark" />
    </main>
  )
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowRight, Code, GitBranch, LayoutDashboard, Play, Route, ShieldCheck, Search } from 'lucide-react'
import { BranchTree } from './BranchTree'
import { CatalogCallout, catalogNotes } from './CatalogBits'
import { SeedChart, SeedMeasurements, TargetEvidence } from './Evidence'
import { RepairStory, repairCount } from './RepairStory'
import { ResultBand } from './ResultChart'
import { ReplayPlayer } from './run/ReplayPlayer'
import { StepGraph } from './run/StepGraph'
import { buildSteps, type StepId } from './run/StepStrip'
import { StepList } from './run/StepList'
import { Fold, TabIntro } from './run/Fold'
import { RunTabs } from './run/RunTabs'
import { VerdictCard } from './run/VerdictCard'
import { PaperHead, PaperByline } from './PaperHead'
import { Film } from './film/Film'
import { DropLast } from './DropLast'
import { ReadmeBadge } from './ReadmeBadge'
import { CommunityFindings, Explanations, communityCount, explanationItems, hasCommunity } from './CommunityCard'
import { FindingExperiments, FindingPreview, experimentCount } from './FindingExperiments'
import { Hint } from './Hint'
import { Trust } from './Trust'
import { CertificateActions } from './CertificateActions'
import { WhatIf } from './WhatIf'
import { useHead } from '../useHead'
import { headFromSummary } from '../paperhead'
import { getCatalogStatement, parseDate, resultSentence, showValue } from '../format'
import { getText, loadRun, runDir, type Json, type RunData, type RunRef } from '../data'
import { OVERALL, RESULT, catalogLabel, de2en, translateClaim } from '../labels'
import { parseTab, tabHash, type HowView, type RunTab } from '../runtabs'
import { tavilySummary } from '../tavilysummary'
import { toPercent } from './ui'

const reduceMotion = () => !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
const INFO = 'The certificate of one check. The paper claimed a number; our agent rebuilt the experiment and an independent program measured it.'
const ICONS: Record<RunTab, React.ReactNode> = {
  overview: <LayoutDashboard className="h-4 w-4" />, how: <Route className="h-4 w-4" />, sources: <Search className="h-4 w-4" />, code: <GitBranch className="h-4 w-4" />, limits: <ShieldCheck className="h-4 w-4" />,
}
const FOLDS_OPEN = ['f-target', 'f-seeds']   // the folds that start open; the others hold long lists

// The certificate of one check. Top: one verdict card. Below: five tabs, so nothing is hidden and nothing needs endless scrolling.
export function RunView({ ref_, initialTab, playTimelapse = false }: { ref_: RunRef; initialTab?: string; playTimelapse?: boolean }) {
  const [run, setRun] = useState<RunData | null>(null)
  const [selected, setSelected] = useState<Json | null>(null)
  const start = parseTab(initialTab)
  const [tab, setTab] = useState<RunTab>(start.tab)
  const [how, setHow] = useState<HowView>(start.how)
  const [file, setFile] = useState<'code' | 'plan'>('code')
  const [open, setOpen] = useState<Set<string>>(() => new Set(FOLDS_OPEN))
  const [timelapse, setTimelapse] = useState(playTimelapse)   // the 30-second panel, opened by the button or by ?play=1
  const barRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const topRef = useRef<HTMLDivElement>(null)
  const filmRef = useRef<HTMLDivElement>(null)
  const stored = useHead(ref_.paper_id)
  useEffect(() => { loadRun(ref_).then(setRun) }, [ref_])
  // ?play=1: bring the whole player into view once it is on the page
  // The scroll container is <main id="app-scroll"> (Layout), not the window; scrollIntoView finds it from the element.
  const [playAsk, setPlayAsk] = useState(0)   // counts ?play=1 links followed while this run is already open
  const playing = timelapse && !!run && run.events.length > 0
  useEffect(() => { if (playing) filmRef.current?.scrollIntoView({ block: 'start' }) }, [playing, playAsk])
  // a link to the same run with another ?tab= (the hash changes, the page stays) switches the tab
  useEffect(() => {
    const f = () => {
      const q = new URLSearchParams(location.hash.split('?')[1] ?? '')
      const n = parseTab(q.get('tab')); setTab(n.tab); setHow(n.how)
      if (q.get('play') === '1') { setTimelapse(true); setPlayAsk((x) => x + 1) }   // "See it in action" while this run is open
    }
    addEventListener('hashchange', f)
    return () => removeEventListener('hashchange', f)
  }, [])
  // code versions of the replay: code_<hash>.py from the export, older exports only have the final code.py
  const finalHash = useMemo(() => {
    const att: Json[] = (run?.dossier?.attempts ?? []).filter((x: Json) => x.measured != null)
    return att.length ? att[att.length - 1].code_hash : null
  }, [run])
  const codeLoader = useCallback((hash: string) => getText(`${runDir(ref_)}code_${hash}.py`).then((t) => t ?? (hash === finalHash ? run?.code ?? null : null)), [ref_, finalHash, run])

  // Bring the top of the panel into view when the reader is further down, so a new tab never opens in the middle.
  const toPanel = (id?: string) => window.setTimeout(() => {
    const el = (id && document.getElementById(id)) || panelRef.current
    if (!el) return
    if (id || el.getBoundingClientRect().top < (barRef.current?.offsetHeight ?? 0)) el.scrollIntoView({ behavior: reduceMotion() ? 'auto' : 'smooth', block: 'start' })
  }, 40)
  const goTab = (t: RunTab, o: { how?: HowView; fold?: string; scroll?: string } = {}) => {
    setTab(t)
    if (o.how) setHow(o.how)
    if (o.fold) setOpen((s) => new Set(s).add(o.fold!))
    try { history.replaceState(null, '', tabHash(location.hash, t)) } catch { /* the URL stays as it was */ }
    toPanel(o.scroll ?? o.fold)
  }
  const setFold = (id: string) => (v: boolean) => setOpen((s) => { if (s.has(id) === v) return s; const n = new Set(s); if (v) n.add(id); else n.delete(id); return n })

  if (!run) return <main className="mx-auto max-w-6xl px-4 py-10 text-muted md:px-8">Loading run data…</main>
  const d = run.dossier
  const head = headFromSummary(d.summary) ?? stored   // what this run wrote itself, else the stored head of the paper
  const claim: Json = d.claims?.[0]
  const v: Json = d.verdicts?.[0]
  if (!claim) return <main className="mx-auto max-w-6xl px-4 py-10 text-muted md:px-8">Empty run.</main>

  const isCatalog = !!ref_.catalog_id   // discovery candidate run through the loop
  const res = v ? (isCatalog ? catalogLabel(v.urteil, ref_.gesamtaussage) : de2en(RESULT, v.urteil)) : { label: 'NO VERDICT', tone: 'gray' as const }
  const fidelityUnchecked = !v?.methode || v.methode === 'nicht geprueft'
  const fmt = (x: number) => showValue(claim.metric, x)
  const fmtSpread = (x: number) => (claim.metric === 'accuracy' ? (x * 100).toFixed(1) : x.toFixed(3))
  const sentence = resultSentence(claim, v, ref_.gesamtaussage)   // one string for the verdict card and the time-lapse
  const steps = buildSteps(run.events, d, { running: false, hasPlan: !!run.plan, metric: claim.metric })
  const notes = isCatalog ? (({ items, warnings }) => items.length + warnings.length)(catalogNotes(run, claim.id)) : 0
  const community = hasCommunity(d), explanations = explanationItems(d, claim.id)
  const nExp = experimentCount(d, claim.id), nRepairs = repairCount(run.events, claim.id)
  const tav = tavilySummary(run.events)
  const paperUrl = claim.target_evidence?.url

  const onStep = (id: StepId) => {
    if (id === 'paper' || id === 'verdict') { goTab('overview'); window.setTimeout(() => topRef.current?.scrollIntoView({ behavior: reduceMotion() ? 'auto' : 'smooth', block: 'start' }), 40) }
    else if (id === 'claim' || id === 'sources') goTab('sources', { fold: 'f-target' })
    else if (id === 'plan') { setFile('plan'); goTab('code', { fold: 'f-plancode' }) }
    else if (id === 'code') { setFile('code'); goTab('code', { fold: 'f-plancode' }) }
    else if (id === 'sandbox') goTab('code', { scroll: 'sec-branches' })
    else if (id === 'check') goTab('how', { how: 'log' })
    else if (id === 'community') goTab('sources', { fold: 'f-community' })
  }

  const h3 = 'm-0 mb-4 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted'
  const toggle = (on: boolean) => `cursor-pointer rounded-lg border px-4 py-2 text-xs font-semibold uppercase tracking-wider ${on ? 'border-line-stronger bg-raised text-fg' : 'border-line bg-transparent text-muted hover:bg-raised-2'}`

  return (
    <main className="mx-auto max-w-6xl px-4 pb-24 md:px-8">
      {timelapse && run.events.length > 0 && (
        <div ref={filmRef} className="scroll-mt-4" data-testid="film"><Film events={run.events} dossier={d} claim={claim} title={run.ref.title} tone={res.tone} resultLabel={res.label} sentence={sentence} fidelityUnchecked={fidelityUnchecked}
          overall={OVERALL[ref_.gesamtaussage ?? ''] ?? null} fmt={fmt} fmtSpread={fmtSpread}
          plan={run.plan} finalCode={run.code} codeLoader={codeLoader} autoplay /></div>
      )}
      <div ref={topRef} className="scroll-mt-4">
        <div className="flex flex-wrap items-center justify-between gap-3 pt-6">
          <a href="#overview" className="inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-muted no-underline transition-colors hover:text-fg-2">
            <ArrowRight className="h-3.5 w-3.5 rotate-180" /> Back to runs
          </a>
          {run.events.length > 0 && !timelapse && (
            <button type="button" onClick={() => setTimelapse(true)} className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-line-strong bg-transparent px-3.5 py-1.5 text-[12px] font-medium text-fg hover:bg-raised/60">
              <Play className="h-3.5 w-3.5" />Play this check in 30 seconds
            </button>
          )}
        </div>
        <header className="pb-6 pt-6 sm:pb-8 sm:pt-8">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">
            {run.ref.paper_id} · {parseDate(run.ref.timestamp)}<Hint text={INFO} />
          </div>
          <h1 className="font-grotesk m-0 mt-3 max-w-4xl text-3xl font-semibold leading-[1.08] text-fg sm:text-4xl">{run.ref.title}</h1>
          <PaperByline head={head} />
        </header>
        <VerdictCard dossier={d} events={run.events} claim={claim} verdict={v} res={res} sentence={sentence} fmt={fmt} fmtSpread={fmtSpread} paperUrl={paperUrl}
          fidelityUnchecked={fidelityUnchecked} gesamt={ref_.gesamtaussage} isCatalog={isCatalog} notes={notes} claimText={translateClaim(claim.text)} onLimits={() => goTab('limits')} onTavily={() => goTab('sources')} />
      </div>

      {v && <CertificateActions run={run} />}
      <RunTabs tab={tab} onTab={(t) => goTab(t)} icons={ICONS} barRef={barRef} />

      <div ref={panelRef} role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} tabIndex={-1} className="scroll-mt-12 pt-10 outline-none">
        {tab === 'overview' && (
          <div className="space-y-12">
            <section aria-label="Claimed against measured">
              <h3 className={h3}>Claimed against measured</h3>
              <ResultBand dossier={d} claimId={claim.id} tone={res.tone} fmt={fmt} fmtSpread={fmtSpread} showKey={false} />
              <p className="mb-0 mt-3 text-[12px] leading-relaxed text-muted">Dots: seeds. Band: claim ± tolerance.</p>
            </section>
            <section aria-label="The nine steps">
              <h3 className={h3}>The nine steps of a check</h3>
              <StepList steps={steps} onStep={onStep} />
              <p className="mb-0 mt-3 text-[12px] text-muted">Click a step to open it.</p>
            </section>
            <section aria-label="Use this check" className="rounded-xl border border-line bg-surface p-6">
              <h3 className={h3}>Before you build on this result</h3>
              <p className="mb-0 mt-3 max-w-2xl text-[14px] leading-relaxed text-fg-4">Choosing a paper to build on? Start with its claim and our measurement above. Open the sources to see where the target came from, then inspect the program and repeated runs. This helps researchers decide what to investigate next. Matching the number still needs a review of the method.</p>
              <div className="mt-5 flex flex-wrap gap-4">
                <button onClick={() => goTab('sources')} className="cursor-pointer border-0 bg-transparent p-0 text-sm text-fg underline underline-offset-4">Inspect sources &amp; setup</button>
                <button onClick={() => goTab('code')} className="cursor-pointer border-0 bg-transparent p-0 text-sm text-fg underline underline-offset-4">Inspect code &amp; measurements</button>
              </div>
            </section>
            {nExp > 0 && (
              <section aria-label="Findings that became experiments">
                <h3 className={h3}>Findings that became experiments</h3>
                <FindingPreview dossier={d} claimId={claim.id} onOpen={() => goTab('sources', { fold: 'f-experiments' })} />
              </section>
            )}
            <WhatIf dossier={d} claim={claim} verdict={v} fmt={fmt} fmtSpread={fmtSpread} />
            {head && <Fold title="About the paper" meta="summary, authors, affiliations" id="f-about" open={open.has('f-about')} onOpenChange={setFold('f-about')}><PaperHead head={head} /></Fold>}
            <a href="#benchmark" className="inline-flex items-center gap-2 text-[13px] font-medium text-fg-3 no-underline hover:text-fg">
              Benchmark: how often Nullius is right<ArrowRight className="h-4 w-4" />
            </a>
          </div>
        )}

        {tab === 'how' && (
          <div>
            <TabIntro title="How we got there" text="The agent’s steps, in order." />
            <div className="mb-6 flex gap-2">
              <button onClick={() => setHow('graph')} aria-pressed={how === 'graph'} className={toggle(how === 'graph')}>Step graph</button>
              <button onClick={() => setHow('log')} aria-pressed={how === 'log'} className={toggle(how === 'log')}>Execution log</button>
            </div>
            {how === 'graph'
              ? <StepGraph events={run.events} running={false} codeLoader={codeLoader} planFull={run.plan} metricHint={claim.metric} />
              : <ReplayPlayer events={run.events} codeLoader={codeLoader} planFull={run.plan} />}
            {nRepairs > 0 && (
              <div className="mt-12">
                <Fold title="Watch the agent fix its own code" meta={`${nRepairs} ${nRepairs === 1 ? 'step' : 'steps'}`} id="f-repairs" open={open.has('f-repairs')} onOpenChange={setFold('f-repairs')}>
                  <RepairStory events={run.events} claimId={claim.id} codeLoader={codeLoader} />
                </Fold>
              </div>
            )}
          </div>
        )}

        {tab === 'sources' && (
          <div>
            <TabIntro title="Sources & Tavily" text="What Tavily found. It never changes the verdict." />
            {tav.total > 0 && (
              <p className="mb-8 mt-0 font-mono text-[12px] text-muted" data-testid="tavily-summary">{tav.total} Tavily {tav.total === 1 ? 'call' : 'calls'} in this run · {tav.parts.join(' · ')}</p>
            )}
            <Fold title="Target value and its quote" meta={`claimed ${fmt(claim.target)}`} id="f-target" open={open.has('f-target')} onOpenChange={setFold('f-target')}>
              <TargetEvidence claim={claim} />
            </Fold>
            {community && (
              <Fold title="What others found" meta={`${communityCount(d)} ${communityCount(d) === 1 ? 'note' : 'notes'}`} id="f-community" open={open.has('f-community')} onOpenChange={setFold('f-community')}>
                <CommunityFindings dossier={d} />
              </Fold>
            )}
            {nExp > 0 && (
              <Fold title="Findings that became experiments" meta={`${nExp} tested`} id="f-experiments" open={open.has('f-experiments')} onOpenChange={setFold('f-experiments')}>
                <FindingExperiments dossier={d} claimId={claim.id} />
              </Fold>
            )}
            {explanations && (
              <Fold title="Possible explanations" meta={`${explanations.length} unverified`} id="f-explain" open={open.has('f-explain')} onOpenChange={setFold('f-explain')}>
                <Explanations items={explanations} />
              </Fold>
            )}
          </div>
        )}

        {tab === 'code' && (
          <div>
            <TabIntro title="Code & branches" text="One sandbox branch per seed, plus plan and code." />
            <section id="sec-branches" className="scroll-mt-16" aria-label="Branch tree">
              <h3 className={h3}>Branch tree</h3>
              <BranchTree dossier={d} claimId={claim.id} events={run.events} onSelect={setSelected} selected={selected?.branch_id ?? null} />
            </section>
            <div className="mt-12">
              <Fold title="Measurements per seed" meta="table and chart" id="f-seeds" open={open.has('f-seeds')} onOpenChange={setFold('f-seeds')}>
                <div className="grid gap-10 lg:grid-cols-2">
                  <SeedMeasurements claimId={claim.id} dossier={d} selected={selected} />
                  <SeedChart claimId={claim.id} dossier={d} />
                </div>
              </Fold>
              <Fold title="Plan and code" meta="exactly as stored" id="f-plancode" open={open.has('f-plancode')} onOpenChange={setFold('f-plancode')}>
                <div className="space-y-4">
                  <div className="flex gap-2">
                    <button onClick={() => setFile('code')} aria-pressed={file === 'code'} className={toggle(file === 'code')}><Code className="mr-2 inline h-3.5 w-3.5" />Agent code</button>
                    <button onClick={() => setFile('plan')} aria-pressed={file === 'plan'} className={toggle(file === 'plan')}>Plan</button>
                  </div>
                  <pre className="m-0 h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-surface p-5 font-mono text-[12px] leading-relaxed text-fg-4">
                    {file === 'code' ? run.code || 'No code snapshot available.' : run.plan || 'No plan recorded for this run.'}
                  </pre>
                </div>
              </Fold>
            </div>
          </div>
        )}

        {tab === 'limits' && (
          <div className="space-y-12">
            <TabIntro title="Limits" text="Why trust the number, and what it does not say." />
            <section aria-label="What the verdict means">
              <h3 className={h3}>What the verdict means</h3>
              <p className="m-0 max-w-3xl text-[14px] leading-relaxed text-fg-3">
                {v ? getCatalogStatement(fmt(claim.target), fmt(v.measured_mean), v.urteil, ref_.gesamtaussage) : `No verdict yet for the claimed ${fmt(claim.target)}.`}
              </p>
              <p className="mb-0 mt-2 max-w-3xl text-[13px] leading-relaxed text-muted">
                Allowed tolerance: ±{claim.metric === 'accuracy' ? toPercent(claim.tolerance) : Number(claim.tolerance).toFixed(3)}, fixed before the run. The verdict uses the mean over all seeds; single seeds may fall outside.
              </p>
            </section>
            {isCatalog && <section aria-label="Notes about this run"><h3 className={h3}>Notes about this run</h3><CatalogCallout run={run} claimId={claim.id} gesamt={ref_.gesamtaussage} /></section>}
            <Trust fidelityUnchecked={fidelityUnchecked} />
            <DropLast dossier={d} claimId={claim.id} />
            <ReadmeBadge ref_={ref_} urteil={v?.urteil} />
          </div>
        )}
      </div>
    </main>
  )
}

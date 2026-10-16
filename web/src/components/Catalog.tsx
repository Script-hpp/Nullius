import { useEffect, useMemo, useState } from 'react'
import { BookOpen, Code, PlayCircle, Play, Loader2, Lock, PauseCircle, Info } from 'lucide-react'
import { Badge, CARD, Chevron, ExtArrow, GoArrow, GoLink, ICON_MD, ICON_SM, INLINE } from './ui'
import { Glossed } from './Glossed'
import { PaperHead } from './PaperHead'
import { useHead } from '../useHead'
import { headFromSummary } from '../paperhead'
import { NextStep, PageIntro } from './page'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, evidenceHref, useCatalog, type CatalogEntry } from '../catalog'
import { catalogLabel, translateReason } from '../labels'
import { cardClaim, noVerdictLabel } from '../catalogCard'
import { startReproduction, useJobInfo, useLiveStatus, useReproductionEligibility, type ReproductionEligibility } from '../live'
import { StartRefused, startBlock, statusFor, type KindStatus } from '../livestatus'
import { referenceShort } from '../referenceBench'
import { loadIndex, type RunRef } from '../data'
import { catalogFilterFromHash, matchesCatalogFilter, recordedDate, type CatalogFilter } from '../catalogFilter'
import { useCollectionPosition } from '../useCollectionPosition'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'

type LiveControl = { available: boolean; busy: boolean; paused: KindStatus | null; starting: boolean; error: string | null; errorIsNotice: boolean; latestLive: string | null; onRun: () => void; eligibility?: ReproductionEligibility }

// One card = one paper claim: verdict, the claim in plain words, claimed against measured, one meta line, one action. Everything else is in "Details".
function CatalogCard({ e, live, samePaper }: { e: CatalogEntry; live: LiveControl; samePaper: boolean }) {
  const v = e.verdict
  const stored = useHead(e.id)
  const head = stored ?? headFromSummary(e.head, e.arxiv_id)   // summaries.json first, else the head the run's dossier carried into catalog.json
  const href = evidenceHref(e)
  const res = v ? catalogLabel(v.urteil, v.gesamtaussage) : null
  const offered = !e.recorded && e.status !== 'nicht_laufbar' && !!e.jury?.ready   // only jury-ready entries get a start control
  const block = startBlock({ available: live.available, kind: live.paused, busy: live.busy, runnable: live.eligibility?.runnable })
  const canRun = offered && live.available && live.eligibility?.runnable === true && !block.blocked
  const date = e.checked_at ? new Date(e.checked_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : null
  const meta = [v ? `${v.seeds} seeds` : null, e.cost_gross_usd != null ? `$${e.cost_gross_usd.toFixed(2)}` : null, date].filter(Boolean).join(' · ')
  const tag = e.recorded ? `Recorded ${recordedDate(e.recorded.date)}` : e.origin === 'reference' ? 'Reference paper' : null
  const why = !v ? (e.status === 'ungeprueft' && !e.reason ? (e.origin === 'reference' ? 'A full run has to finish before this entry shows a result.' : 'The claim was read from the paper and has not been re-run.') : `${e.reason ? translateReason(e.reason) : STATUS_TEXT[e.status]}`) : null
  return (
    <article className={`relative flex h-full min-w-0 flex-col rounded-xl border border-line bg-surface p-6 ${CARD}`}>
      <div className="flex min-h-6 items-center justify-between gap-3">
        {res ? <CatalogVerdict entry={e} /> : <Badge tone="gray">{noVerdictLabel(e.status, !!e.reason)}</Badge>}
        {tag && <span className="shrink-0 text-[11px] text-muted">{tag}</span>}
      </div>
      <h3 className="font-grotesk m-0 mt-4 text-balance text-xl font-semibold leading-snug tracking-[-0.02em] text-fg">
        {href ? <a href={href} className="text-inherit no-underline after:absolute after:inset-0 after:rounded-xl focus-visible:underline focus-visible:underline-offset-4">{cardClaim(e)}</a> : cardClaim(e)}
      </h3>
      <p className="m-0 mt-1.5 line-clamp-1 text-[13px] text-muted" title={e.title}>{e.title}</p>
      {v ? (
        <div className="mt-5"><ClaimedMeasured entry={e} /></div>
      ) : (
        <p className="mb-0 mt-5 line-clamp-3 text-[13px] leading-relaxed text-fg-4">{why}</p>
      )}
      {meta && <p className="m-0 mt-auto pt-5 text-[12px] text-muted">{meta}</p>}

      <div className={`relative z-10 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line/60 pt-4 ${meta ? 'mt-4' : 'mt-auto'}`}>
        {href ? (
          <a href={href} className={`nx-go ${INLINE} rounded-full border border-fg px-4 py-1.5 text-[13px] font-medium text-fg no-underline transition-colors hover:bg-fg hover:text-bg`}>Open result<GoArrow className="h-3.5 w-3.5 shrink-0" /></a>
        ) : (
          <a href={`https://arxiv.org/abs/${e.arxiv_id}`} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} rounded-full border border-line-strong px-4 py-1.5 text-[13px] font-medium text-fg-2 no-underline transition-colors hover:bg-raised`}>Read the paper<ExtArrow /></a>
        )}
        {canRun && (
          <button onClick={live.onRun} disabled={live.starting} className={`${INLINE} cursor-pointer rounded-full border border-line-strong bg-transparent px-4 py-1.5 text-[13px] font-medium text-fg-2 transition-colors hover:bg-raised disabled:cursor-wait disabled:opacity-60`}>
            {live.starting ? <Loader2 className={`${ICON_MD} animate-spin`} /> : <Play className={ICON_MD} />}Start a check
          </button>
        )}
        {offered && block.blocked && live.available && <span className={`${INLINE} text-[12px] text-muted`} title={block.long}><Lock className={ICON_SM} aria-hidden />{block.short}</span>}
        {e.recorded && <span className={`${INLINE} text-[12px] text-muted`} title={e.recorded.note}><Lock className={ICON_SM} aria-hidden />No new check: recorded result</span>}
        {live.error && <span className={`w-full text-[12px] ${live.errorIsNotice ? 'text-muted' : 'text-tone-red'}`}>{live.error}{href && <> <a href={href} className="text-fg-3 underline underline-offset-2 hover:text-fg">Open the recorded result</a></>}</span>}
      </div>

      <details className="relative z-10 mt-4 text-[12px] text-muted">
        <summary className={`nx-summary ${INLINE} cursor-pointer hover:text-fg-3`}>Details<Chevron className="h-3.5 w-3.5" /></summary>
        <div className="mt-3 space-y-3 leading-relaxed">
          {samePaper && <p className="m-0">Same paper, separate dataset check. This result is for {e.task ?? e.dataset}.</p>}
          {e.recorded && <p className="m-0">{e.recorded.summary ? `${e.recorded.summary} ` : ''}{e.recorded.note}</p>}
          {e.status === 'ungeprueft' && e.reason && <p className="m-0 text-tone-amber">Last check stopped: {translateReason(e.reason)}</p>}
          {!e.recorded && live.eligibility?.runnable === false && <p className="m-0">No live check for this entry: {translateReason(live.eligibility.reason ?? '')}</p>}
          {e.origin === 'reference' && e.benchmark && <p className="m-0">{referenceShort(e.benchmark)} <GoLink href="#benchmark" className="text-fg-3 hover:text-fg-max">Benchmark results</GoLink></p>}
          {v && <p className="m-0">Tolerance <span className="font-mono text-fg-3">±{v.tolerance}</span>{!!v.graph_repairs && <> · Graph repairs <span className="font-mono text-fg-3">{v.graph_repairs}</span></>}</p>}
          <PaperHead head={head} compact />
          <p className="m-0 flex flex-wrap items-center gap-x-4 gap-y-1.5">
            {e.run_id && <EvidenceLink entry={e} tab="replay"><PlayCircle className={ICON_MD} />30-second replay</EvidenceLink>}
            {e.run_id && <EvidenceLink entry={e} tab="log">Event log</EvidenceLink>}
            {live.latestLive && <GoLink href={`#/run/${live.latestLive}`} className="text-fg-3 hover:text-fg-max">Latest live run</GoLink>}
            {e.github_url && <a href={e.github_url} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}><Code className={ICON_MD} />Authors’ repository<ExtArrow /></a>}
            <a href={`https://arxiv.org/abs/${e.arxiv_id}`} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}><BookOpen className={ICON_MD} />arXiv<ExtArrow /></a>
          </p>
        </div>
      </details>
    </article>
  )
}

export function Catalog() {
  const entries = useCatalog()
  const { entries: eligibility } = useReproductionEligibility()
  const { available, ready: jobReady, info, refresh } = useJobInfo()
  // Availability changes the intro's height, so restore only after it is known.
  useCollectionPosition(entries !== null && jobReady)
  const { status, refresh: refreshStatus } = useLiveStatus()
  const repro = statusFor(status, 'reproduction')
  const banner = startBlock({ available, kind: repro, busy: !!info?.busy })
  const [starting, setStarting] = useState<string | null>(null)
  const [error, setError] = useState<{ id: string; msg: string; notice: boolean } | null>(null)
  const [runs, setRuns] = useState<RunRef[]>([])
  const [filter, setFilter] = useState<CatalogFilter>(() => catalogFilterFromHash(location.hash))
  useEffect(() => {
    const sync = () => setFilter(catalogFilterFromHash(location.hash))
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  }, [])
  const chooseFilter = (view: CatalogFilter) => {
    setFilter(view)
    history.replaceState(null, '', `#catalog${view === 'featured' ? '' : `?view=${view}`}`)
  }
  useEffect(() => { loadIndex().then(setRuns) }, [])
  const latestLive = (id: string) => runs.filter((r) => r.live && r.catalog_id === id).map((r) => r.id).sort().pop() ?? null
  const run = async (id: string) => {
    setStarting(id); setError(null)
    try {
      const jobId = await startReproduction(id, true)
      window.location.assign(`#/live/${jobId}`)
    } catch (err) {
      setError({ id, msg: err instanceof Error ? err.message : 'Could not start the run.', notice: err instanceof StartRefused && err.reason !== null })
      refresh(); refreshStatus()
    } finally { setStarting(null) }
  }
  const shown = useMemo(() => {
    const rank = (e: CatalogEntry) => {
      if (!e.verdict) return 5
      const tone = catalogLabel(e.verdict.urteil, e.verdict.gesamtaussage).tone
      return ({ green: 0, amber: 1, red: 2, violet: 3, gray: 4 })[tone]
    }
    return (entries ?? [])
      .filter((e) => matchesCatalogFilter(e, filter))
      .sort((a, b) => rank(a) - rank(b) || (b.checked_at ?? '').localeCompare(a.checked_at ?? ''))
  }, [entries, filter])
  const total = entries?.length ?? 0

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <PageIntro kicker="Catalog" title="Papers we have checked"
        sub="Each card is one paper claim: what it says, what we measured."
        about={<>
          <p className="m-0">Featured shows checks with published evidence. All entries adds failed checks and candidates that were never run. A failed check does not prove the paper wrong.</p>
          <ol className="m-0 list-none space-y-1 p-0">
            <li><span className="font-medium text-fg">1 Collect.</span> Reference papers and Tavily finds with a quotable claim.</li>
            <li><span className="font-medium text-fg">2 One full run.</span> Rebuilt by the agent, measured by the harness.</li>
            <li><span className="font-medium text-fg">3 Listed.</span> A finished check with measurements and evidence.</li>
          </ol>
          <p className="m-0">Supported datasets: <Glossed text="Cora, CiteSeer (cited articles, accuracy) and ETTh1 (transformer temperature, 96-hour forecast, MSE)." /> Another dataset needs a task and harness first. This demo has no paper-URL input. A live check can differ from the stored result: the agent writes new code each time.</p>
        </>} />

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div role="group" aria-label="Filter catalog" className="flex flex-wrap gap-2">
          {([{ value: 'featured', label: 'Featured' }, { value: 'all', label: 'All' }, { value: 'failed', label: 'Failed' }, { value: 'unchecked', label: 'Unchecked' }] as const).map((item) => (
            <button key={item.value} type="button" aria-pressed={filter === item.value} onClick={() => chooseFilter(item.value)} className={`cursor-pointer rounded-full border px-3.5 py-1.5 text-[13px] font-medium ${filter === item.value ? 'border-fg bg-inverse text-on-inverse' : 'border-line-strong bg-transparent text-fg-3 hover:bg-raised'}`}>{item.label}</button>
          ))}
        </div>
        <span className={label}>{entries ? `${shown.length} of ${total} shown` : 'Loading'}</span>
      </div>
      <p className="mb-6 mt-0 flex items-start gap-2 text-[13px] leading-relaxed text-muted">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>Measured with our setup. Whether the code follows the paper’s method is not checked; the info icon on a card says so.</span>
      </p>
      {available && banner.blocked && (
        <p role="status" className="mb-6 mt-0 flex w-fit max-w-full items-start gap-2 rounded-lg border border-line bg-surface px-4 py-3 text-[13px] leading-relaxed text-fg-3">
          <PauseCircle className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
          <span>{banner.long}</span>
        </p>
      )}
      {!available && jobReady && <p className="mb-6 mt-0 text-[13px] leading-relaxed text-muted">Live checks are off on this site. Recorded results and replays open as usual.</p>}

      <div className="grid gap-5 md:grid-cols-2">
        {shown.map((e) => (
          <CatalogCard key={e.id} e={e} samePaper={(entries ?? []).filter((other) => other.arxiv_id === e.arxiv_id).length > 1} live={{
            available, busy: !!info?.busy, paused: repro, starting: starting === e.id, error: error?.id === e.id ? error.msg : null, errorIsNotice: !!error?.notice,
            latestLive: latestLive(e.id), onRun: () => run(e.id), eligibility: eligibility?.[e.id],
          }} />
        ))}
      </div>
      {entries && !shown.length && (
        <div className="rounded-xl border border-line bg-surface p-6 text-[13px] text-muted">
          <Badge tone="gray">Nothing to show</Badge>
          <p className="mt-2">No entries match this filter. <GoLink href="#catalog?view=all">View all entries</GoLink></p>
        </div>
      )}
      {(() => {
        const first = shown.map((e) => ({ e, href: evidenceHref(e) })).find((x) => x.href)
        return first
          ? <NextStep title="Read the evidence" text={`Open the run of "${first.e.title}".`} label="Open the evidence" href={first.href!} />
          : <NextStep title="Find more papers" text="Nothing to open yet." label="Discover papers" href="#discover" />
      })()}
    </main>
  )
}

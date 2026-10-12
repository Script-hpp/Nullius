import { useEffect, useMemo, useState } from 'react'
import { BookOpen, Code, CheckCircle2, CircleSlash, PlayCircle, Play, Loader2, Library, PauseCircle } from 'lucide-react'
import { Badge, CARD_HOVER, Chevron, ExtArrow, GoLink, ICON_MD, INLINE } from './ui'
import { Glossed } from './Glossed'
import { PaperHead } from './PaperHead'
import { useHead } from '../useHead'
import { headFromSummary } from '../paperhead'
import { NextStep, PageIntro } from './page'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, evidenceHref, useCatalog, type CatalogEntry } from '../catalog'
import { catalogLabel, translateJuryReason, translateReason } from '../labels'
import { startReproduction, useJobInfo, useLiveStatus } from '../live'
import { StartRefused, isPaused, pausedText, resumeText, statusFor } from '../livestatus'
import { referenceShort } from '../referenceBench'
import { loadIndex, type RunRef } from '../data'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'

type LiveControl = { available: boolean; busy: boolean; paused: boolean; starting: boolean; error: string | null; errorIsNotice: boolean; latestLive: string | null; onRun: () => void }

function CatalogCard({ e, live, samePaper }: { e: CatalogEntry; live: LiveControl; samePaper: boolean }) {
  const v = e.verdict
  const stored = useHead(e.id)
  const head = stored ?? headFromSummary(e.head, e.arxiv_id)   // summaries.json first, else the head the run's dossier carried into catalog.json
  const canRun = live.available && e.status !== 'nicht_laufbar'
  const why = live.paused ? 'Live checks are paused' : live.busy ? 'Another run is active' : undefined
    const href = evidenceHref(e)
  return (
    <article className={`relative flex h-full flex-col rounded-xl border border-line bg-surface p-6 ${href ? `${CARD_HOVER} focus-within:border-line-strong` : ''}`}>
      <p className="m-0 mb-3 text-sm font-semibold text-fg">Dataset check: {e.task ?? e.dataset ?? 'Not specified'}</p>
      <div className="flex items-start justify-between gap-4">
        <h3 className="font-grotesk m-0 text-xl font-semibold leading-snug tracking-[-0.02em] text-fg">
          {href ? <a href={href} className="text-inherit no-underline after:absolute after:inset-0 after:rounded-xl focus-visible:underline focus-visible:underline-offset-4">{e.title}</a> : e.title}
        </h3>
        <span className="shrink-0 font-mono text-[10px] text-muted">{e.arxiv_id}</span>
      </div>
      {samePaper && <p className="mb-0 mt-2 text-[12px] leading-relaxed text-muted">Same paper, separate dataset check. This result is for {e.task ?? e.dataset}.</p>}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted">
        <span>Task <span className="text-fg-3">{e.task ?? '–'}</span></span>
        <span>Claim <span className="text-fg-3">{e.claim.method ?? '–'}</span></span>
        <span>{STATUS_TEXT[e.status]}</span>
      </div>
      {e.origin !== 'reference' && e.checked_at && (
        <p className="m-0 mt-3 text-[12px] text-muted">Checked · {new Date(e.checked_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
      )}
      {e.origin === 'reference' && (
        <div className="mt-3 space-y-1">
          <Badge tone="gray" icon={<Library className="h-4 w-4 shrink-0" />} title="A paper from our benchmark with a known reference result">Reference paper</Badge>
          {e.benchmark && (
            <p className="m-0 text-[12px] leading-relaxed text-muted">
              {referenceShort(e.benchmark)} <GoLink href="#benchmark" className="relative z-10 text-fg-3 hover:text-fg-max">Benchmark results</GoLink>
            </p>
          )}
        </div>
      )}
      <div className="relative z-10"><PaperHead head={head} compact /></div>

      {v ? (
        <div className="mt-4 space-y-4">
          <CatalogVerdict entry={e} />
          <ClaimedMeasured entry={e} />
        </div>
      ) : (
        <p className="mt-4 text-[12px] leading-relaxed text-muted">
          {e.status === 'ungeprueft' ? (e.origin === 'reference' ? 'Not run through the catalog loop yet: a full run has to finish before this entry can show a result here.' : 'Unverified candidate: the claim was read from the paper, not re-run.') : `${STATUS_TEXT[e.status]}${e.reason ? `: ${translateReason(e.reason)}` : ''}`}
        </p>
      )}

      {/* Stats and run controls move as one block, centered in the space left above the footer. */}
      <div className="flex flex-1 flex-col justify-center gap-3 py-5">
        {v && (
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-muted">
            <span>Seeds <span className="font-mono text-fg-3">{v.seeds}</span></span>
            <span>Tolerance <span className="font-mono text-fg-3">±{v.tolerance}</span></span>
            {e.cost_gross_usd != null && <span>Cost <span className="font-mono text-fg-3">${e.cost_gross_usd.toFixed(3)}</span></span>}
            {!!v.graph_repairs && <span>Graph repairs <span className="font-mono text-fg-3">{v.graph_repairs}</span></span>}
          </div>
        )}
        {canRun && (
          <div className="relative z-10 flex flex-wrap items-center gap-x-4 gap-y-2">
            <button onClick={live.onRun} disabled={!!why || live.starting} title={why}
              className={`${INLINE} rounded-lg border border-line-strong px-3 py-1.5 text-[12px] font-medium leading-none text-fg hover:bg-raised transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed`}>
              {live.starting ? <Loader2 className={`${ICON_MD} animate-spin`} /> : <Play className={ICON_MD} />}Run reproduction
            </button>
            <span className="text-[11px] text-muted">Quick check, 3 seeds, watched live</span>
            {live.latestLive && <GoLink href={`#/run/${live.latestLive}`} className="text-[11px] text-fg-3 hover:text-fg-max">Latest live run</GoLink>}
            {live.error && <span className={`w-full text-[12px] ${live.errorIsNotice ? 'text-muted' : 'text-tone-red'}`}>{live.error}</span>}
          </div>
        )}
      </div>

      <div className="relative z-10 flex flex-wrap items-center justify-between gap-3 border-t border-line/60 pt-3 text-[11px] leading-none">
        <span className={`${INLINE} text-fg-4`} title={e.jury ? translateJuryReason(e.jury.reason) : undefined}>
          {e.jury?.ready && e.run_id ? <CheckCircle2 className={ICON_MD} /> : <CircleSlash className={ICON_MD} />}
          {v && !e.run_id ? 'Evidence not published here' : e.jury ? (e.jury.ready ? 'Jury ready' : `Not jury ready: ${translateJuryReason(e.jury.reason)}`) : 'Not jury ready: not run yet'}
        </span>
        <span className="flex flex-wrap items-center gap-4">
          <EvidenceLink entry={e}>View evidence</EvidenceLink>
          <EvidenceLink entry={e} tab="log"><PlayCircle className={ICON_MD} />Watch replay</EvidenceLink>
          {e.github_url && (
            <a href={e.github_url} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}><Code className={ICON_MD} />Code<ExtArrow /></a>
          )}
          <a href={`https://arxiv.org/abs/${e.arxiv_id}`} target="_blank" rel="noreferrer" className={`nx-ext ${INLINE} text-fg-3 hover:text-fg-max`}><BookOpen className={ICON_MD} />arXiv<ExtArrow /></a>
        </span>
      </div>
    </article>
  )
}

export function Catalog() {
  const entries = useCatalog()
  const { available, info, refresh } = useJobInfo()
  const { status, refresh: refreshStatus } = useLiveStatus()
  const repro = statusFor(status, 'reproduction')
  const paused = isPaused(repro)
  const [starting, setStarting] = useState<string | null>(null)
  const [error, setError] = useState<{ id: string; msg: string; notice: boolean } | null>(null)
  const [runs, setRuns] = useState<RunRef[]>([])
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
      .filter((e) => e.jury?.ready && e.run_id)
      .sort((a, b) => rank(a) - rank(b) || (b.checked_at ?? '').localeCompare(a.checked_at ?? ''))
  }, [entries])
  const total = entries?.length ?? 0

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <PageIntro kicker="Catalog" title="Papers we have checked"
        what="Finished checks with measurements and published evidence. Open a result to inspect its evidence."
        look="Open an entry marked Jury ready, or start a quick check." />

      <details className="group mb-10 border-t border-line pt-6">
        <summary className={`nx-summary ${INLINE} cursor-pointer text-[11px] font-semibold uppercase tracking-[0.2em] text-muted hover:text-fg-3`}>How this catalog is made · supported datasets<Chevron className="h-3.5 w-3.5" /></summary>
        <div className="mt-6 grid gap-10 md:grid-cols-2 md:gap-0">
          <div className="md:pr-12">
            <ol className="m-0 list-none space-y-3 p-0 text-[15px] leading-relaxed text-fg-4">
              <li className="flex gap-3"><span className="shrink-0 font-mono text-fg-3">01</span><span><span className="font-medium text-fg">Collect.</span> Reference papers and Tavily finds with a quotable claim.</span></li>
              <li className="flex gap-3"><span className="shrink-0 font-mono text-fg-3">02</span><span><span className="font-medium text-fg">One full run.</span> Rebuilt by the agent, measured by the harness.</span></li>
              <li className="flex gap-3"><span className="shrink-0 font-mono text-fg-3">03</span><span><span className="font-medium text-fg">Listed.</span> Jury ready: a finished check with measurements and evidence to explore. Method fidelity is not verified.</span></li>
            </ol>
            <p className="mb-0 mt-4 text-[13px] leading-relaxed text-muted">A live check can differ from the stored result: the agent writes new code each time.</p>
          </div>
          <div className="border-line md:border-l md:pl-12">
            <dl className="m-0 space-y-2 text-[15px] leading-relaxed text-fg-4">
              <div><dt className="inline font-medium text-fg">Cora</dt><dd className="m-0 inline"> <Glossed text="Cited scientific articles, seven topics, accuracy." /></dd></div>
              <div><dt className="inline font-medium text-fg">Cora-LCC</dt><dd className="m-0 inline"> Largest connected group, random 20/30/rest splits.</dd></div>
              <div><dt className="inline font-medium text-fg">CiteSeer</dt><dd className="m-0 inline"> <Glossed text="Cited scientific articles, six topics, accuracy." /></dd></div>
              <div><dt className="inline font-medium text-fg">ETTh1</dt><dd className="m-0 inline"> <Glossed text="Transformer temperature, 96-hour forecast, MSE." /></dd></div>
            </dl>
            <p className="mb-0 mt-4 text-[13px] leading-relaxed text-muted">Another dataset needs a task and harness first.</p>
          </div>
        </div>
      </details>

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <span className="text-sm text-fg-3">Jury-ready papers</span>
        {available && info?.busy && (
          <span className="text-[12px] text-tone-amber">A {info.kind === 'discovery' ? 'discovery' : 'reproduction'} run is active. Live runs are one at a time.</span>
        )}
        <span className={label}>{entries ? `${shown.length} of ${total} shown · Reproduced first` : 'Loading'}</span>
      </div>
      <p className="mb-6 mt-0 max-w-3xl text-[13px] leading-relaxed text-muted">
        {available ? 'Live checks need no sign-in.' : 'Live checks are off here; finished runs and replays are open.'}
      </p>
      {available && paused && (
        <p role="status" className="mb-6 mt-0 flex w-fit max-w-full items-center gap-2 rounded-lg border border-line bg-surface px-4 py-3 text-[13px] leading-relaxed text-fg-3">
          <PauseCircle className="h-4 w-4 shrink-0 text-muted" />
          <span>{pausedText(repro?.reason)} {resumeText(repro?.reason, repro?.reset_at)}</span>
        </p>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        {shown.map((e) => (
          <CatalogCard key={e.id} e={e} samePaper={(entries ?? []).filter((other) => other.arxiv_id === e.arxiv_id).length > 1} live={{
            available, busy: !!info?.busy, paused, starting: starting === e.id, error: error?.id === e.id ? error.msg : null, errorIsNotice: !!error?.notice,
            latestLive: latestLive(e.id), onRun: () => run(e.id),
          }} />
        ))}
      </div>
      {entries && !shown.length && (
        <div className="rounded-xl border border-line bg-surface p-6 text-[13px] text-muted">
          <Badge tone="gray">Nothing to show</Badge>
          <p className="mt-2">No jury-ready checks have been published yet.</p>
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

import { useEffect, useMemo, useState } from 'react'
import { BookOpen, Code, CheckCircle2, CircleSlash, PlayCircle, Play, Loader2 } from 'lucide-react'
import { Badge } from './ui'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, useCatalog, type CatalogEntry } from '../catalog'
import { translateJuryReason, translateReason } from '../labels'
import { startReproduction, useJobInfo } from '../live'
import { loadIndex, type RunRef } from '../data'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'

type LiveControl = { available: boolean; busy: boolean; hasCode: boolean; starting: boolean; error: string | null; latestLive: string | null; onRun: () => void }

function CatalogCard({ e, live }: { e: CatalogEntry; live: LiveControl }) {
  const v = e.verdict
  const canRun = live.available && e.status !== 'nicht_laufbar'
  const why = live.busy ? 'Another run is active' : !live.hasCode ? 'Enter the access code first' : undefined
  return (
    <article className="rounded-xl border border-line bg-panel/50 p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-[15px] font-semibold leading-snug text-fg">{e.title}</h3>
        <span className="shrink-0 font-mono text-[10px] text-muted">{e.arxiv_id}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted">
        <span>Task <span className="text-fg-3">{e.task ?? '–'}</span></span>
        <span>Claim <span className="text-fg-3">{e.claim.method ?? '–'}</span></span>
        <span>{STATUS_TEXT[e.status]}</span>
      </div>

      {v ? (
        <div className="mt-4 space-y-4">
          <CatalogVerdict entry={e} />
          <ClaimedMeasured entry={e} />
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-muted">
            <span>Seeds <span className="font-mono text-fg-3">{v.seeds}</span></span>
            <span>Tolerance <span className="font-mono text-fg-3">±{v.tolerance}</span></span>
            {e.cost_gross_usd != null && <span>Cost <span className="font-mono text-fg-3">${e.cost_gross_usd.toFixed(3)}</span></span>}
            {!!v.graph_repairs && <span>Graph repairs <span className="font-mono text-fg-3">{v.graph_repairs}</span></span>}
          </div>
        </div>
      ) : (
        <p className="mt-4 text-[12px] leading-relaxed text-muted">
          {e.status === 'ungeprueft' ? 'Unverified candidate: the claim was read from the paper, not re-run.' : `${STATUS_TEXT[e.status]}${e.reason ? `: ${translateReason(e.reason)}` : ''}`}
        </p>
      )}

      {canRun && (
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <button onClick={live.onRun} disabled={!!why || live.starting} title={why}
            className="inline-flex items-center gap-2 rounded-lg border border-line-strong px-3 py-1.5 text-[12px] font-medium text-fg hover:bg-raised transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed">
            {live.starting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}Run reproduction
          </button>
          <span className="text-[11px] text-muted">Quick check, 3 seeds, watched live</span>
          {live.latestLive && <a href={`#/run/${live.latestLive}`} className="inline-flex items-center gap-1 text-[11px] text-fg-3 hover:text-fg-max">Latest live run</a>}
          {live.error && <span className="w-full text-[12px] text-tone-red">{live.error}</span>}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line/60 pt-3 text-[11px]">
        <span className="inline-flex items-center gap-1.5 text-fg-4" title={e.jury ? translateJuryReason(e.jury.reason) : undefined}>
          {e.jury?.ready ? <CheckCircle2 className="w-3.5 h-3.5" /> : <CircleSlash className="w-3.5 h-3.5" />}
          {e.jury ? (e.jury.ready ? 'Jury ready' : `Not jury ready: ${translateJuryReason(e.jury.reason)}`) : 'Not jury ready: not run yet'}
        </span>
        <span className="flex flex-wrap items-center gap-4">
          <EvidenceLink entry={e}>View evidence</EvidenceLink>
          <EvidenceLink entry={e} tab="log"><PlayCircle className="w-3.5 h-3.5" />Watch replay</EvidenceLink>
          {e.github_url && (
            <a href={e.github_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-3 hover:text-fg-max"><Code className="w-3.5 h-3.5" />Code</a>
          )}
          <a href={`https://arxiv.org/abs/${e.arxiv_id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-3 hover:text-fg-max"><BookOpen className="w-3.5 h-3.5" />arXiv</a>
        </span>
      </div>
    </article>
  )
}

export function Catalog() {
  const entries = useCatalog()
  const [juryOnly, setJuryOnly] = useState(true)
  const { available, info, refresh } = useJobInfo()
  const [code, setCode] = useState('')
  const [starting, setStarting] = useState<string | null>(null)
  const [error, setError] = useState<{ id: string; msg: string } | null>(null)
  const [runs, setRuns] = useState<RunRef[]>([])
  useEffect(() => { loadIndex().then(setRuns) }, [])
  const latestLive = (id: string) => runs.filter((r) => r.live && r.catalog_id === id).map((r) => r.id).sort().pop() ?? null
  const run = async (id: string) => {
    setStarting(id); setError(null)
    try {
      const jobId = await startReproduction(id, code, true)
      window.location.assign(`#/live/${jobId}`)
    } catch (err) {
      setError({ id, msg: err instanceof Error ? err.message : 'Could not start the run.' })
      refresh()
    } finally { setStarting(null) }
  }
  const shown = useMemo(() => (entries ?? []).filter((e) => !juryOnly || e.jury?.ready), [entries, juryOnly])
  const total = entries?.length ?? 0

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-fg">Catalog</h1>
        <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-fg-4">
          Papers found by Tavily discovery and run once through the full reproduction loop. Results are measured with our setup and are preliminary until the method fidelity is checked.
        </p>
      </div>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <label className="inline-flex items-center gap-2 text-sm text-fg-3 cursor-pointer">
          <input type="checkbox" checked={juryOnly} onChange={(ev) => setJuryOnly(ev.target.checked)} className="accent-fg-3" />
          Jury-ready only
        </label>
        {available && (
          <div className="flex flex-wrap items-center gap-3">
            <label htmlFor="repro-code" className={label}>Access code</label>
            <input id="repro-code" type="password" autoComplete="off" value={code} onChange={(ev) => setCode(ev.target.value)}
              className="block w-40 rounded-lg border border-line bg-raised px-3 py-1.5 font-mono text-sm text-fg outline-none focus:border-line-stronger" />
            {info?.busy && <span className="text-[12px] text-tone-amber">A {info.kind === 'discovery' ? 'discovery' : 'reproduction'} run is active. Live runs are one at a time.</span>}
          </div>
        )}
        <span className={label}>{entries ? `${shown.length} of ${total} shown` : 'Loading'}</span>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {shown.map((e) => (
          <CatalogCard key={e.id} e={e} live={{
            available, busy: !!info?.busy, hasCode: !!code, starting: starting === e.id, error: error?.id === e.id ? error.msg : null,
            latestLive: latestLive(e.id), onRun: () => run(e.id),
          }} />
        ))}
      </div>
      {entries && !shown.length && (
        <div className="rounded-xl border border-line bg-panel/50 p-6 text-[13px] text-muted">
          <Badge tone="gray">Nothing to show</Badge>
          <p className="mt-2">{total ? 'No entry is jury ready yet. Turn the filter off to see every candidate and why it is held back.' : 'The catalog is empty.'}</p>
        </div>
      )}
    </main>
  )
}

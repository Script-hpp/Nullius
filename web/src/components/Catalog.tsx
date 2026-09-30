import { useMemo, useState } from 'react'
import { BookOpen, Code, CheckCircle2, CircleSlash, PlayCircle } from 'lucide-react'
import { Badge } from './ui'
import { CatalogVerdict, ClaimedMeasured, EvidenceLink } from './CatalogBits'
import { STATUS_TEXT, useCatalog, type CatalogEntry } from '../catalog'

const label = 'text-[10px] font-semibold uppercase tracking-widest text-zinc-500'

function CatalogCard({ e }: { e: CatalogEntry }) {
  const v = e.verdict
  return (
    <article className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-5">
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-[15px] font-semibold leading-snug text-zinc-100">{e.title}</h3>
        <span className="shrink-0 font-mono text-[10px] text-zinc-500">{e.arxiv_id}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-zinc-500">
        <span>Task <span className="text-zinc-300">{e.task ?? '–'}</span></span>
        <span>Claim <span className="text-zinc-300">{e.claim.method ?? '–'}</span></span>
        <span>{STATUS_TEXT[e.status]}</span>
      </div>

      {v ? (
        <div className="mt-4 space-y-4">
          <CatalogVerdict entry={e} />
          <ClaimedMeasured entry={e} />
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-zinc-500">
            <span>Seeds <span className="font-mono text-zinc-300">{v.seeds}</span></span>
            <span>Tolerance <span className="font-mono text-zinc-300">±{v.tolerance}</span></span>
            {e.cost_gross_usd != null && <span>Cost <span className="font-mono text-zinc-300">${e.cost_gross_usd.toFixed(3)}</span></span>}
            {!!v.graph_repairs && <span>Graph repairs <span className="font-mono text-zinc-300">{v.graph_repairs}</span></span>}
          </div>
        </div>
      ) : (
        <p className="mt-4 text-[12px] leading-relaxed text-zinc-500">
          {e.status === 'ungeprueft' ? 'Unverified candidate: the claim was read from the paper, not re-run.' : `${STATUS_TEXT[e.status]}${e.reason ? `: ${e.reason}` : ''}`}
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800/60 pt-3 text-[11px]">
        <span className="inline-flex items-center gap-1.5 text-zinc-400" title={e.jury?.reason}>
          {e.jury?.ready ? <CheckCircle2 className="w-3.5 h-3.5" /> : <CircleSlash className="w-3.5 h-3.5" />}
          {e.jury ? (e.jury.ready ? 'Jury ready' : `Not jury ready: ${e.jury.reason}`) : 'Not jury ready: not run yet'}
        </span>
        <span className="flex flex-wrap items-center gap-4">
          <EvidenceLink entry={e}>View evidence</EvidenceLink>
          <EvidenceLink entry={e} tab="log"><PlayCircle className="w-3.5 h-3.5" />Watch replay</EvidenceLink>
          {e.github_url && (
            <a href={e.github_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-zinc-300 hover:text-white"><Code className="w-3.5 h-3.5" />Code</a>
          )}
          <a href={`https://arxiv.org/abs/${e.arxiv_id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-zinc-300 hover:text-white"><BookOpen className="w-3.5 h-3.5" />arXiv</a>
        </span>
      </div>
    </article>
  )
}

export function Catalog() {
  const entries = useCatalog()
  const [juryOnly, setJuryOnly] = useState(true)
  const shown = useMemo(() => (entries ?? []).filter((e) => !juryOnly || e.jury?.ready), [entries, juryOnly])
  const total = entries?.length ?? 0

  return (
    <main className="mx-auto max-w-6xl px-4 md:px-8 pb-20">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Catalog</h1>
        <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-zinc-400">
          Papers found by Tavily discovery and run once through the full reproduction loop. Results are measured with our setup and are preliminary until the method fidelity is checked.
        </p>
      </div>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <label className="inline-flex items-center gap-2 text-sm text-zinc-300 cursor-pointer">
          <input type="checkbox" checked={juryOnly} onChange={(ev) => setJuryOnly(ev.target.checked)} className="accent-zinc-300" />
          Jury-ready only
        </label>
        <span className={label}>{entries ? `${shown.length} of ${total} shown` : 'Loading'}</span>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {shown.map((e) => <CatalogCard key={e.id} e={e} />)}
      </div>
      {entries && !shown.length && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-6 text-[13px] text-zinc-500">
          <Badge tone="gray">Nothing to show</Badge>
          <p className="mt-2">{total ? 'No entry is jury ready yet. Turn the filter off to see every candidate and why it is held back.' : 'The catalog is empty.'}</p>
        </div>
      )}
    </main>
  )
}

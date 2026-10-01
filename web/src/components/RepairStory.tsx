import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, Check, ExternalLink, GitCompareArrows, Layers, Search, TriangleAlert, Wrench, X } from 'lucide-react'
import { highlight } from './run/PythonCode'
import { buildStory, lineDiff, type Card, type Hunk } from '../repairStory'
import type { Event } from '../data'

// "Watch the agent fix its own code": the repairs of one claim as cards, in order. Everything shown comes from the run's events and its
// stored code versions; the diff is computed here from the two stored files.
type Loader = (hash: string) => Promise<string | null>

function Diff({ from, to, codeLoader }: { from: string; to: string; codeLoader: Loader }) {
  const [state, setState] = useState<{ hunks: Hunk[] } | 'missing' | null>(null)
  const [all, setAll] = useState(false)
  useEffect(() => {
    let live = true
    Promise.all([codeLoader(from), codeLoader(to)]).then(([a, b]) => { if (live) setState(a != null && b != null ? { hunks: lineDiff(a, b) } : 'missing') })
    return () => { live = false }
  }, [from, to, codeLoader])
  if (state === null) return <p className="m-0 text-[12px] text-muted">Loading the code versions…</p>
  if (state === 'missing') return <p className="m-0 text-[12px] text-muted">The code files of these two versions are not part of this export, so there is no diff to show.</p>
  const { hunks } = state
  if (!hunks.length) return <p className="m-0 text-[12px] text-muted">The two versions are identical.</p>
  const total = hunks.reduce((n, h) => n + h.length, 0)
  const shown = all ? hunks : hunks.reduce<Hunk[]>((acc, h) => (acc.reduce((n, x) => n + x.length, 0) < 40 ? [...acc, h] : acc), [])
  return (
    <div className="min-w-0 rounded-lg border border-line bg-surface">
      <div className="max-h-[26rem] overflow-auto py-2 font-mono text-[11.5px] leading-relaxed">
        {shown.map((h, i) => (
          <div key={i} className={i > 0 ? 'mt-2 border-t border-dashed border-line pt-2' : ''}>
            {h.map((l, k) => (
              <div key={k} className={`flex whitespace-pre px-3 ${l.t === 'add' ? 'bg-tone-green/10' : l.t === 'del' ? 'bg-tone-red/10' : ''}`}>
                <span aria-hidden className={`w-4 shrink-0 select-none ${l.t === 'add' ? 'text-tone-green' : l.t === 'del' ? 'text-tone-red' : 'text-muted'}`}>{l.t === 'add' ? '+' : l.t === 'del' ? '-' : ' '}</span>
                <code className={l.t === 'ctx' ? 'text-fg-4' : 'text-fg-2'}>{highlight(l.text)}</code>
              </div>
            ))}
          </div>
        ))}
      </div>
      {shown.length < hunks.length && (
        <button onClick={() => setAll(true)} className="w-full cursor-pointer border-0 border-t border-line bg-transparent px-3 py-1.5 text-left text-[11px] text-muted hover:text-fg-3">Show all {total} changed and nearby lines</button>
      )}
    </div>
  )
}

function Shell({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <li className="relative min-w-0 border-l border-line pb-10 pl-6 last:pb-0 md:pl-8">
      <span className="absolute -left-[13px] top-0 flex h-6 w-6 items-center justify-center rounded-full border border-line bg-surface text-fg-4">{icon}</span>
      <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted">{label}</div>
      {children}
    </li>
  )
}

function CardView({ c, codeLoader }: { c: Card; codeLoader: Loader }) {
  if (c.kind === 'candidates') {
    return <Shell icon={<Layers className="h-3.5 w-3.5" />} label="Several versions"><p className="mb-0 mt-2 max-w-3xl text-[15px] leading-relaxed text-fg-3">{c.text}</p></Shell>
  }
  if (c.kind === 'repo_diff') {
    return (
      <Shell icon={<GitCompareArrows className="h-3.5 w-3.5" />} label="Compared with the paper&rsquo;s repository">
        <p className="mb-0 mt-2 max-w-3xl text-[15px] leading-relaxed text-fg-3">{c.text}</p>
        {c.fromHash && c.toHash && c.fromHash !== c.toHash && <div className="mt-4"><Diff from={c.fromHash} to={c.toHash} codeLoader={codeLoader} /></div>}
      </Shell>
    )
  }
  const label = c.toVersion ? `Version ${c.fromVersion} to version ${c.toVersion}` : `After version ${c.fromVersion}`
  const ok = c.outcome.tone === 'ok'
  return (
    <Shell icon={<Wrench className="h-3.5 w-3.5" />} label={label}>
      <div className="mt-2 flex items-start gap-2 text-[15px] font-medium leading-snug text-fg-3"><TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-tone-amber" />{c.problemTitle}</div>
      <pre className="m-0 mt-2 max-w-full overflow-x-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-raised px-3 py-2 font-mono text-[12px] leading-relaxed text-fg-4">{c.problem}</pre>
      {c.search && (
        <div className="mt-4 min-w-0">
          <div className="flex items-start gap-2 text-[13px] leading-relaxed text-fg-4"><Search className="mt-0.5 h-3.5 w-3.5 shrink-0" /><span className="min-w-0 break-words">Tavily search for the error: <span className="font-mono text-[12px] text-fg-3">{c.search.query}</span></span></div>
          {c.search.links.length > 0
            ? <ul className="m-0 mt-2 flex list-none flex-wrap gap-x-5 gap-y-1 p-0 pl-5 text-[12.5px]">{c.search.links.map((l) => (
                <li key={l.url}><a href={l.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-4 underline decoration-line-strong underline-offset-4 hover:text-fg">{l.domain}<ExternalLink className="h-3 w-3" /></a></li>))}</ul>
            : <p className="m-0 mt-1 pl-5 text-[12.5px] text-muted">The search found nothing usable.</p>}
        </div>
      )}
      {c.toHash && c.fromHash !== c.toHash && <div className="mt-4"><Diff from={c.fromHash} to={c.toHash} codeLoader={codeLoader} /></div>}
      <div className={`mt-4 flex items-start gap-2 text-[14px] leading-relaxed ${ok ? 'text-tone-green' : c.outcome.tone === 'fail' ? 'text-tone-red' : 'text-muted'}`}>
        {ok ? <Check className="mt-0.5 h-4 w-4 shrink-0" /> : c.outcome.tone === 'fail' ? <X className="mt-0.5 h-4 w-4 shrink-0" /> : <ArrowRight className="mt-0.5 h-4 w-4 shrink-0" />}
        <span className="min-w-0 break-words">{c.outcome.text}</span>
      </div>
    </Shell>
  )
}

export function RepairStory({ events, claimId, codeLoader }: { events: Event[]; claimId: string; codeLoader: Loader }) {
  const cards = useMemo(() => buildStory(events, claimId), [events, claimId])
  if (!cards.length) return null
  return (
    <section id="sec-repairs" className="scroll-mt-4 min-w-0" aria-label="How the agent fixed its own code">
      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Watch the agent fix its own code</div>
      <p className="mb-0 mt-4 max-w-3xl text-[15px] leading-relaxed text-fg-3">Our agent is allowed to fail. Here is how it found and fixed its own mistakes in this run.</p>
      <ol className="m-0 mt-8 list-none p-0">
        {cards.map((c, i) => <CardView key={i} c={c} codeLoader={codeLoader} />)}
      </ol>
    </section>
  )
}

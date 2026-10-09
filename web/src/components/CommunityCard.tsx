import { ExternalLink } from 'lucide-react'
import { Badge } from './ui'
import type { Json } from '../data'
import type { Tone } from '../labels'

const KIND: Record<string, { label: string; tone: Tone }> = {
  'protocol note': { label: 'Protocol note', tone: 'amber' },
  'reproduction report': { label: 'Reproduction report', tone: 'violet' },
  other: { label: 'Note', tone: 'gray' },
}
const REPO_CHECK: Record<string, string> = { bestaetigt: 'Confirmed by the repository code', widerlegt: 'Contradicted by the repository code' }

const host = (url: string) => { try { const u = new URL(url); return u.host + u.pathname.replace(/\/$/, '') } catch { return url } }
const when = (iso: string | undefined) => { const d = iso ? new Date(iso) : null; return d && !Number.isNaN(d.getTime()) ? d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : null }

function Quote({ text, url }: { text: string; url: string }) {
  return (
    <>
      <blockquote className="m-0 mt-2 border-l-2 border-line-strong/60 py-1 pl-3 font-mono text-[12px] italic leading-relaxed text-fg-4">“{text}”</blockquote>
      <a className="mt-2 inline-flex max-w-full items-center gap-1.5 text-[11px] text-tone-blue transition-colors hover:text-tone-blue-hover" href={url} target="_blank" rel="noreferrer">
        <ExternalLink className="h-3 w-3 shrink-0" /><span className="truncate">{host(url)}</span>
      </a>
    </>
  )
}

export const communityCount = (dossier: Json) => ((dossier.community?.findings ?? []) as Json[]).length
export const hasCommunity = (dossier: Json) => !!dossier.community
export const explanationItems = (dossier: Json, claimId: string): Json[] | null => {
  const ex = (dossier.explanations ?? {})[claimId] as Json | undefined
  return ex ? ((ex.items ?? []) as Json[]) : null
}

// Words next to measurement: what the public says about the paper. Reported only, never part of the verdict.
export function CommunityFindings({ dossier }: { dossier: Json }) {
  const c = dossier.community as Json | null | undefined
  if (!c) return null
  const findings: Json[] = c.findings ?? []
  return (
    <div data-testid="community-card">
      <p className="m-0 mb-4 max-w-2xl text-[13px] leading-relaxed text-fg-4">Public notes about this paper from GitHub issues, reviews and the web, each with a quote from the page. They sit next to our measurement and never change it.</p>
      {findings.length ? (
        <ul className="m-0 list-none p-0">
          {findings.map((f, i) => (
            <li key={i} className="border-t border-line py-5 first:border-t-0 first:pt-0">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <Badge tone={(KIND[f.kind] ?? KIND.other).tone}>{(KIND[f.kind] ?? KIND.other).label}</Badge>
                {REPO_CHECK[f.repo_check] && <span className="font-mono text-[11px] text-muted">{REPO_CHECK[f.repo_check]}</span>}
              </div>
              <p className="m-0 text-[14px] font-medium leading-relaxed text-fg-2">{f.statement}</p>
              <Quote text={f.quote} url={f.url} />
            </li>
          ))}
        </ul>
      ) : (
        <p className="m-0 text-sm italic text-muted">No public discussion found.</p>
      )}
      <p className="mb-0 mt-4 text-[11px] leading-relaxed text-subtle">
        {[when(c.queried_at) && `Searched ${when(c.queried_at)}${c.mode === 'cache' ? ' (saved result, valid 7 days)' : ''}`,
          `${(c.pages ?? []).length} page${(c.pages ?? []).length === 1 ? '' : 's'} read`,
          c.dropped ? `${c.dropped} statement${c.dropped === 1 ? '' : 's'} dropped because the quote is not in the page` : null].filter(Boolean).join(' · ')}
      </p>
    </div>
  )
}

// Why our number may differ from the paper, according to public sources. Unverified.
export function Explanations({ items }: { items: Json[] }) {
  return (
    <div>
      <p className="m-0 mb-4 max-w-2xl text-[13px] leading-relaxed text-fg-4">Why our number may differ from the paper, according to public sources. We have not tested any of these.</p>
      {items.length ? (
        <ul className="m-0 list-none p-0">
          {items.map((it, i) => (
            <li key={i} className="border-t border-line py-5 first:border-t-0 first:pt-0">
              <div className="mb-2"><Badge tone="gray">Unverified hypothesis</Badge></div>
              <p className="m-0 text-[14px] font-medium leading-relaxed text-fg-2">{it.explanation}</p>
              <Quote text={it.quote} url={it.url} />
            </li>
          ))}
        </ul>
      ) : (
        <p className="m-0 text-sm italic text-muted">No explanation found in public sources.</p>
      )}
    </div>
  )
}

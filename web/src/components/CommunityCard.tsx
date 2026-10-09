import { ExternalLink, Lightbulb, MessagesSquare } from 'lucide-react'
import { Badge, Section } from './ui'
import { FindingExperiments } from './FindingExperiments'
import type { Json } from '../data'
import type { Tone } from '../labels'
import { checksFor } from '../objections'

const KIND: Record<string, { label: string; tone: Tone }> = {
  'protocol note': { label: 'Protocol note', tone: 'amber' },
  'reproduction report': { label: 'Reproduction report', tone: 'violet' },
  other: { label: 'Note', tone: 'gray' },
}
const REPO_CHECK: Record<string, string> = { bestaetigt: 'Confirmed by the repository code', widerlegt: 'Contradicted by the repository code' }

const host = (url: string) => { try { const u = new URL(url); return u.host + u.pathname.replace(/\/$/, '') } catch { return url } }
const when = (iso: string | undefined) => { const d = iso ? new Date(iso) : null; return d && !Number.isNaN(d.getTime()) ? d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : null }

// Words next to measurement: what the public says about the paper. Reported only, never part of the verdict.
export function CommunityCard({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const c = dossier.community as Json | null | undefined
  const ex = (dossier.explanations ?? {})[claimId] as Json | undefined
  const checks = checksFor(dossier, claimId)
  if (!c && !ex && !checks.length) return null   // runs without a community check (older runs, benchmark): no card
  const findings: Json[] = c?.findings ?? []
  const items: Json[] = ex?.items ?? []
  return (
    <div className="space-y-6" data-testid="community-card">
      {c && (
        <Section title="What others found" hint="Reported only · not part of the verdict" className="min-w-0">
          <div className="mb-5 flex items-start gap-3 text-[12px] leading-relaxed text-fg-4">
            <MessagesSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
            <p>Public notes about this paper from GitHub issues, reviews and the web, each with a quote from the page. They sit next to our measurement and never change it.</p>
          </div>
          {findings.length ? (
            <ul className="space-y-4">
              {findings.map((f, i) => (
                <li key={i} className="rounded-lg border border-line/50 bg-surface p-4">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <Badge tone={(KIND[f.kind] ?? KIND.other).tone}>{(KIND[f.kind] ?? KIND.other).label}</Badge>
                    {REPO_CHECK[f.repo_check] && <span className="font-mono text-[11px] text-muted">{REPO_CHECK[f.repo_check]}</span>}
                  </div>
                  <p className="text-[13px] font-medium leading-relaxed text-fg-2">{f.statement}</p>
                  <blockquote className="mt-2 rounded-md border-l-2 border-line-strong/60 bg-panel/40 px-3 py-2 font-mono text-[12px] italic leading-relaxed text-fg-4">“{f.quote}”</blockquote>
                  <a className="mt-2 inline-flex items-center gap-1.5 text-[11px] text-tone-blue transition-colors hover:text-tone-blue-hover" href={f.url} target="_blank" rel="noreferrer">
                    <ExternalLink className="h-3 w-3" />
                    <span className="max-w-[320px] truncate">{host(f.url)}</span>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm italic text-muted">No public discussion found.</p>
          )}
          <p className="mt-4 text-[11px] leading-relaxed text-subtle">
            {[when(c.queried_at) && `Searched ${when(c.queried_at)}${c.mode === 'cache' ? ' (saved result, valid 7 days)' : ''}`,
              `${(c.pages ?? []).length} page${(c.pages ?? []).length === 1 ? '' : 's'} read`,
              c.dropped ? `${c.dropped} statement${c.dropped === 1 ? '' : 's'} dropped because the quote is not in the page` : null].filter(Boolean).join(' · ')}
          </p>
        </Section>
      )}
      <FindingExperiments dossier={dossier} claimId={claimId} />
      {ex && (
        <Section title="Possible explanations" hint="Unverified hypotheses" className="min-w-0">
          <div className="mb-5 flex items-start gap-3 text-[12px] leading-relaxed text-fg-4">
            <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-muted" />
            <p>Why our number may differ from the paper, according to public sources. We have not tested any of these.</p>
          </div>
          {items.length ? (
            <ul className="space-y-4">
              {items.map((it, i) => (
                <li key={i} className="rounded-lg border border-line/50 bg-surface p-4">
                  <div className="mb-2"><Badge tone="gray">Unverified hypothesis</Badge></div>
                  <p className="text-[13px] font-medium leading-relaxed text-fg-2">{it.explanation}</p>
                  <blockquote className="mt-2 rounded-md border-l-2 border-line-strong/60 bg-panel/40 px-3 py-2 font-mono text-[12px] italic leading-relaxed text-fg-4">“{it.quote}”</blockquote>
                  <a className="mt-2 inline-flex items-center gap-1.5 text-[11px] text-tone-blue transition-colors hover:text-tone-blue-hover" href={it.url} target="_blank" rel="noreferrer">
                    <ExternalLink className="h-3 w-3" />
                    <span className="max-w-[320px] truncate">{host(it.url)}</span>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm italic text-muted">No explanation found in public sources.</p>
          )}
        </Section>
      )}
    </div>
  )
}

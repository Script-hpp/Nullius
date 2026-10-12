import type { Json } from './data'

export function sourceUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) ? url.href : null
  } catch { return null }
}

// A recorded source is useful even when automatic target extraction failed.
// Reading it never upgrades the stored target or column-verification status.
export function targetSource(claim: Json, sources: Json[] = []) {
  const ev = claim?.target_evidence
  const recorded = String(claim?.source ?? '').match(/^(?:discovery:\s*)?(https?:\/\/[^\s,]+)(?:,\s*(?:Zitat|Quote):\s*([\s\S]+))?$/)
  const paper = sources.find((s) => s.kind === 'paper' && sourceUrl(s.url))
  const url = sourceUrl(ev?.url) ?? sourceUrl(recorded?.[1]) ??
    (claim?.evidence_urls ?? []).map(sourceUrl).find((u: string | null) => u && /(^|\.)arxiv\.org$/.test(new URL(u).hostname)) ?? sourceUrl(paper?.url)
  return {
    url,
    quote: typeof ev?.quote === 'string' && ev.quote ? ev.quote : recorded?.[2] ?? null,
    table: typeof ev?.table === 'string' ? ev.table : null,
    extracted: !!ev,
    columnVerified: ev?.column_check === 'ok',
    recordedLocation: typeof claim?.source === 'string' && !recorded ? claim.source : null,
  }
}

export function setupSourceUrl(source: string, dossier: Json, claim: Json): string | null {
  const direct = sourceUrl(source)
  if (direct) return direct
  if (source === 'paper' || source === 'discovery') return targetSource(claim, dossier.sources).url
  const file = source.match(/^repo:(.+)@([a-f0-9]{7,40})$/i)
  if (!file && source !== 'repo') return null
  const repo = sourceUrl(dossier.repo?.url)
  if (!repo) return null
  if (!file) return repo
  const url = new URL(repo)
  if (url.hostname !== 'github.com' || file[1].split('/').some((p) => !p || p === '.' || p === '..')) return repo
  const path = url.pathname.replace(/\.git\/?$/, '').replace(/\/$/, '')
  return `${url.origin}${path}/blob/${file[2]}/${file[1].split('/').map(encodeURIComponent).join('/')}`
}

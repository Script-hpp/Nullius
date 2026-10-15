import type { Json } from './data'
import type { CatalogEntry } from './catalog'

export type DEvent = { type: string; ts: number; seq?: number; [k: string]: Json }

export function parseLiveEvent(data: string, now = Date.now() / 1000): DEvent | null {
  try {
    const event = JSON.parse(data)
    if (!event || typeof event.type !== 'string' || !event.type) return null
    return { ...event, ts: typeof event.ts === 'number' && Number.isFinite(event.ts) ? event.ts : now }
  } catch { return null }
}

export type PaperState = 'reading' | 'claim' | 'searching' | 'candidate' | 'rejected'

export type DClaim = {
  method?: string
  metric?: string
  target?: number
  horizon?: number
  input_length?: number | null
  quote?: string
  matched_task?: string | null
  protocol?: string
  column_check?: 'ok' | 'unchecked'
}

export type DPaper = {
  id: string
  title: string
  dataset?: string
  state: PaperState
  stage?: string
  reason?: string
  claim?: DClaim
  communityHits?: number
  protocol?: string
  foundBy?: string
  cached?: boolean
  githubUrl?: string | null
  catalogId?: string
  runnable?: boolean
  runReason?: string | null
  /** Readiness check after discovery: target table, authors' code and setup, before any experiment is offered. */
  readiness?: Readiness
}

export type Readiness = { state: 'checking' | 'ready' | 'blocked'; reason?: string | null; stage?: string | null }

function readinessOf(value: unknown): Readiness | undefined {
  if (!value || typeof value !== 'object') return undefined
  const r = value as Record<string, unknown>
  const state = r.state === 'checking' || r.state === 'ready' || r.state === 'blocked' ? r.state
    : typeof r.ready === 'boolean' ? (r.ready ? 'ready' : 'blocked') : undefined
  if (!state) return undefined
  return { state, reason: typeof r.reason === 'string' ? r.reason : null, stage: typeof r.stage === 'string' ? r.stage : null }
}

export type FeedCall = {
  seq: number
  purpose?: string
  endpoint: 'search' | 'extract'
  query: string
  hits: number
  durationMs: number
  topHits: { url: string; title: string }[]
}

// A saved experiment belongs to a particular claim, not just an arXiv paper.
// Discovery itself reads claims; it does not produce reproduction measurements.
export function savedCheckForClaim(paper: DPaper, entries: CatalogEntry[]): CatalogEntry | undefined {
  const claim = paper.claim
  if (!claim) return undefined
  const name = (value?: string) => (value ?? '').toLowerCase().replace(/[\s_-]/g, '')
  const metric = (value?: string) => value?.toLowerCase() === 'acc' ? 'accuracy' : value?.toLowerCase()
  const target = (value: number | undefined, kind: string | undefined) =>
    typeof value === 'number' && Number.isFinite(value) ? (kind === 'accuracy' && value > 1 ? value / 100 : value) : null
  const kind = metric(claim.metric)
  const value = target(claim.target, kind)
  const task = name(claim.matched_task ?? paper.dataset)
  if (!kind || value === null || !task || !name(claim.method)) return undefined
  return entries.filter((entry) => {
    const stored = entry.claim
    const other = target(stored.target, metric(stored.metric))
    return entry.arxiv_id === paper.id && name(stored.method) === name(claim.method) &&
      name(entry.task ?? entry.dataset) === task && metric(stored.metric) === kind &&
      other !== null && Math.abs(other - value) < 1e-8 &&
      (kind === 'accuracy' || claim.horizon != null && stored.horizon === claim.horizon) &&
      (claim.input_length == null || stored.input_length === claim.input_length)
  }).sort((a, b) => (b.checked_at ?? '').localeCompare(a.checked_at ?? ''))[0]
}

export type TimelineItem = { kind: 'tavily' | 'llm'; start: number; end: number; label: string }

export type DiscoveryView = {
  feed: FeedCall[]
  papers: DPaper[]
  tavilyCalls: number
  llmCalls: number
  cachedLlmCalls: number
  costUsd: number
  funnel: { label: string; value: number; note?: string }[]
  timeline: TimelineItem[]
  t0: number
  tEnd: number
  activeId: string | null
  notes: { kind: 'warning' | 'error'; message: string }[]
}

const ARXIV_RE = /arxiv\.org\/(?:abs|html|pdf)\/(\d{4}\.\d{4,5})/

export const arxivIdOf = (url: string | undefined): string | null => {
  const m = ARXIV_RE.exec(url ?? '')
  return m ? m[1] : null
}

export function parseClaim(text: unknown): DClaim | undefined {
  if (typeof text !== 'string') return undefined
  try {
    const c = JSON.parse(text)
    return typeof c === 'object' && c ? c : undefined
  } catch {
    return undefined
  }
}

// Plain-English wording for the technical rejection reasons of the discovery pipeline.
export function humanReject(stage: string | undefined, reason: string | undefined): string {
  const r = reason ?? ''
  const table: [RegExp, string][] = [
    [/pre-filter/i, 'The abstract does not mention the benchmark together with a result.'],
    [/benchmark not found in readable source/i, 'The readable paper source does not mention this benchmark.'],
    [/no results found/i, 'No papers were found for this benchmark.'],
    [/volltext nicht lesbar/i, 'Full text could not be read.'],
    [/llm_truncated/i, 'Model answer was cut off (not counted against the paper).'],
    [/nicht lesbar/i, 'Model answer could not be read (not counted against the paper).'],
    [/no numerical claim/i, 'No checkable number found in the protocol and results sections.'],
    [/no quote provided/i, 'The model gave no supporting quote.'],
    [/quote not found/i, 'The quoted sentence does not appear in the paper text.'],
    [/method missing/i, 'The method behind the number is not named in the quote or the table header.'],
    [/cited prior work/i, 'The selected method is an earlier model cited by this paper. Discovery needs a result for this paper’s own proposed method.'],
    [/target missing/i, 'The reported number could not be parsed.'],
    [/target value not found/i, 'The number is not contained in the quote.'],
    [/metrik nicht messbar/i, 'The metric cannot be measured by our tasks (for example a relative improvement).'],
    [/another column/i, 'The number sits in another column of the results table, not in the proposed method column.'],
    [/not cpu-feasible/i, 'Needs pretraining, a foundation model or external data; we only re-run methods that train from scratch on CPU.'],
    [/horizont nicht|prediction horizon could not be verified/i, 'The extraction could not confirm a result for prediction horizon 96. This does not establish that the paper has no compatible result.'],
  ]
  for (const [re, text] of table) if (re.test(r)) return text
  if (stage === 'protocol') return r || 'The paper uses a different evaluation protocol.'
  return r || 'Rejected.'
}

export const STAGE_LABEL: Record<string, string> = {
  prefilter: 'Prefilter',
  extract: 'Claim extraction',
  protocol: 'Protocol check',
  quote: 'Quote check',
  number: 'Number check',
  feasibility: 'Feasibility check',
}

export function deriveView(events: DEvent[]): DiscoveryView {
  const feed: FeedCall[] = []
  const papers = new Map<string, DPaper>()
  const titles = new Map<string, string>()
  const timeline: TimelineItem[] = []
  const notes: DiscoveryView['notes'] = []
  const seen = new Set<string>()
  const foundIds = new Set<string>()
  let cachedCandidates = 0
  let legacySearch = true
  let llmCalls = 0
  let cachedLlmCalls = 0
  let costUsd = 0
  let tavilyCalls = 0
  const rejects: Record<string, number> = { prefilter: 0, extract: 0, protocol: 0, quote: 0, number: 0, feasibility: 0 }
  let extractingCount = 0
  let candidates = 0
  let activeId: string | null = null

  const paper = (id: string): DPaper => {
    let p = papers.get(id)
    if (!p) {
      p = { id, title: titles.get(id) ?? `arXiv:${id}`, state: 'reading' }
      papers.set(id, p)
    }
    return p
  }

  for (const e of events) {
    if (e.type === 'tavily_call') {
      tavilyCalls++
      const hits = Number(e.hits) || 0
      const dur = Number(e.duration_ms) || 0
      const endpoint = e.endpoint === 'extract' ? 'extract' : 'search'
      const topHits = (e.top_hits ?? []).filter((h: Json) => h?.url)
      feed.push({ seq: e.seq ?? feed.length, purpose: e.purpose, endpoint, query: String(e.query ?? ''), hits, durationMs: dur, topHits })
      if (Number.isFinite(e.ts)) timeline.push({ kind: 'tavily', start: e.ts - dur / 1000, end: e.ts, label: `Tavily ${endpoint}` })
      if (e.purpose === 'find_papers' && Array.isArray(e.arxiv_ids)) {
        legacySearch = false
        for (const id of e.arxiv_ids) foundIds.add(id)
      }
      if (endpoint === 'search') {
        for (const h of topHits) {
          const id = arxivIdOf(h.url)
          if (id && h.title && !titles.has(id)) titles.set(id, h.title)
          if (id && legacySearch) seen.add(id)
        }
      } else {
        const id = arxivIdOf(e.query)
        const p = id ? papers.get(id) : undefined
        if (p && p.state === 'reading') p.state = 'claim'
      }
    } else if (e.type === 'llm_call') {
      llmCalls++
      if (e.cached) cachedLlmCalls++
      costUsd += e.cached ? 0 : Number(e.cost_usd) || 0
      const dur = Number(e.duration_ms) || 0
      if (Number.isFinite(e.ts)) timeline.push({ kind: 'llm', start: e.ts - dur / 1000, end: e.ts, label: e.cached ? 'Model call (cached)' : 'Model call' })
    } else if (e.type === 'extracting') {
      extractingCount++
      seen.add(e.arxiv_id)
      const p = paper(e.arxiv_id)
      p.title = e.title ?? p.title
      p.dataset = e.dataset
      p.state = 'reading'
      activeId = e.arxiv_id
    } else if (e.type === 'searching') {
      const p = paper(e.arxiv_id)
      p.state = 'searching'
      p.claim = parseClaim(e.claim?.text)
      activeId = e.arxiv_id
    } else if (e.type === 'candidate' && typeof e.duplicate_of === 'string') {
      // The same claim was found before: one card, one start button. The earlier find is the offer.
      const open = papers.get(e.arxiv_id as string)
      if (open && open.state !== 'candidate') papers.delete(e.arxiv_id as string)
      if (activeId === e.arxiv_id) activeId = null
    } else if (e.type === 'candidate') {
      candidates++
      if (e.cached) { cachedCandidates++; seen.add(e.arxiv_id) }
      const p = paper(e.arxiv_id)
      // Keep distinct claims from one paper when loading saved live candidates.
      if (p.state === 'candidate' && p.catalogId && p.catalogId !== e.catalog_id) {
        papers.set(p.catalogId, { ...p })
      }
      if (e.title) p.title = e.title
      if (e.dataset) p.dataset = e.dataset
      if (e.claim && typeof e.claim === 'object') p.claim = e.claim
      p.catalogId = typeof e.catalog_id === 'string' ? e.catalog_id : undefined
      p.runnable = typeof e.runnable === 'boolean' ? e.runnable : undefined
      p.runReason = typeof e.run_reason === 'string' ? e.run_reason : null
      p.readiness = readinessOf(e.readiness)
      p.cached = !!e.cached
      if ('github_url' in e) p.githubUrl = e.github_url
      p.state = 'candidate'
      p.reason = e.reason
      p.communityHits = e.community_hits
      p.protocol = e.protocol
      p.foundBy = e.found_by
      if (activeId === e.arxiv_id) activeId = null
    } else if (e.type === 'readiness') {
      const target = [...papers.values()].find((x) => x.catalogId === e.catalog_id)
      if (target) target.readiness = readinessOf(e)
    } else if (e.type === 'reject') {
      const stage = String(e.stage ?? '')
      if (stage in rejects) rejects[stage]++
      if (e.arxiv_id) {
        seen.add(e.arxiv_id)
        const p = paper(e.arxiv_id)
        if (e.title) p.title = e.title
        if (e.dataset) p.dataset = e.dataset
        p.state = 'rejected'
        p.stage = stage
        p.reason = e.reason
        if (activeId === e.arxiv_id) activeId = null
      }
    } else if (e.type === 'warning' || e.type === 'error') {
      notes.push({ kind: e.type, message: String(e.message ?? '') })
    }
  }

  for (const p of papers.values()) if (p.title.startsWith('arXiv:') && titles.has(p.id)) p.title = titles.get(p.id)!

  // Found = unique arXiv ids of the paper searches only (repo and community searches excluded); older recordings: all ids seen.
  const found = Math.max(foundIds.size, seen.size)
  // Examined = papers whose source was read (up to 3 model reads per benchmark); prefilter rejects are only a note, not a funnel drop.
  const examined = extractingCount + cachedCandidates
  const withClaim = Math.max(examined - rejects.extract - rejects.quote - rejects.number, 0)
  const unclear = [...papers.values()].filter(
    (p) => (p.state === 'searching' || p.state === 'candidate') && (p.protocol ?? p.claim?.protocol) === 'unklar').length
  const confirmed = Math.max(withClaim - rejects.protocol - unclear, 0)
  const funnel = [
    { label: 'Found by Tavily', value: found },
    { label: 'Examined', value: examined,
      note: rejects.prefilter ? `${rejects.prefilter} dropped by the benchmark filter` : undefined },
    { label: 'Claim with quote', value: withClaim },
    { label: 'Protocol confirmed', value: confirmed, note: unclear ? `${unclear} kept as candidate, protocol not stated` : undefined },
    { label: 'Candidate', value: candidates },
  ]

  const times = events.map((e) => e.ts).filter((ts) => typeof ts === 'number' && Number.isFinite(ts))
  const t0 = times.length ? Math.min(times[0], ...timeline.map((t) => t.start)) : 0
  const tEnd = times.length ? times[times.length - 1] : 0
  return {
    feed, papers: [...papers.values()], tavilyCalls, llmCalls, cachedLlmCalls, costUsd, funnel, timeline, t0, tEnd, activeId, notes,
  }
}

// Accuracy is a share and shown as percent; MSE and MAE are plain decimals (never converted to percent).
export function formatMetric(metric: string | undefined, target: number | undefined): string {
  if (target === undefined || target === null || Number.isNaN(Number(target))) return '–'
  const v = Number(target)
  if (metric === 'accuracy') return `${(v <= 1 ? v * 100 : v).toFixed(2)}%`
  return `${(metric ?? '').toUpperCase()} ${v.toFixed(3)}`.trim()
}

// Short excerpt of a quote around the reported number: [before, number, after].
export function quoteExcerpt(quote: string | undefined, target: number | undefined, radius = 90): [string, string, string] {
  const q = (quote ?? '').replace(/\s*\n\s*/g, ' ').replace(/\s+/g, ' ').trim()
  if (!q) return ['', '', '']
  if (target !== undefined) {
    for (const m of q.matchAll(/\d+(?:\.\d+)?/g)) {
      const n = Number(m[0])
      if (Math.abs(n - target) < 1e-9 || Math.abs(n / 100 - target) < 1e-9 || Math.abs(n * 100 - target) < 1e-9) {
        const i = m.index ?? 0
        const from = Math.max(0, i - radius)
        const to = Math.min(q.length, i + m[0].length + radius)
        return [(from > 0 ? '… ' : '') + q.slice(from, i), m[0], q.slice(i + m[0].length, to) + (to < q.length ? ' …' : '')]
      }
    }
  }
  return [q.length > 2 * radius ? q.slice(0, 2 * radius) + ' …' : q, '', '']
}

export const domainOf = (url: string): string => {
  try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url }
}

export const shorten = (s: string, n = 90): string => (s.length > n ? s.slice(0, n - 1) + '…' : s)

export function parseJsonl(text: string): DEvent[] {
  return text.split('\n').filter(Boolean).flatMap((l) => {
    try { return [JSON.parse(l) as DEvent] } catch { return [] }
  })
}


export type QuoteCells = { head: string[]; before: string[]; cell: [string, string, string]; after: string[]; all: string[]; hasGap: boolean }

// Table quotes: no pipes in the UI. Show row head plus two cells before/after the number; the full row is collapsible.
export function quoteCells(quote: string | undefined, target: number | undefined): QuoteCells | null {
  if (!quote || !quote.includes('|')) return null
  const lines = quote.split('\n').filter((l) => l.includes('|'))
  const isTarget = (t: string) => {
    for (const m of t.matchAll(/\d+(?:\.\d+)?/g)) {
      const n = Number(m[0])
      if (target !== undefined && (Math.abs(n - target) < 1e-9 || Math.abs(n / 100 - target) < 1e-9 || Math.abs(n * 100 - target) < 1e-9)) return true
    }
    return false
  }
  const rows = lines.map((l) => l.split('|').map((c) => c.replace(/\s+/g, ' ').trim()).filter(Boolean))
  const row = rows.find((r) => r.some(isTarget)) ?? rows[rows.length - 1] ?? []
  const hi = row.findIndex(isTarget)
  if (hi < 0) return { head: [], before: [], cell: ['', '', ''], after: [], all: row, hasGap: false }
  const from = Math.max(0, hi - 2)
  const head = from > 1 ? [row[0]] : []
  const [b, n, a] = quoteExcerpt(row[hi], target, 40)
  return {
    head, before: row.slice(from, hi), cell: [b, n, a], after: row.slice(hi + 1, hi + 3), all: row,
    hasGap: from > head.length || hi + 3 < row.length,
  }
}

// Only an unreadable full text is a reading problem; every other extract-stage reject is about the claim.
export function stageLabel(stage: string | undefined, reason: string | undefined): string {
  if (stage === 'extract' && /volltext nicht lesbar/i.test(reason ?? '')) return 'Reading'
  return STAGE_LABEL[stage ?? ''] ?? stage ?? ''
}

const ETT_TASKS = ['ETTh1']

const TASK_TEXT: Record<string, string> = {
  Cora: 'Cora · Planetoid split',
  'Cora-LCC': 'Cora · random splits, largest component',
  CiteSeer: 'CiteSeer · Planetoid split',
}

export function benchmarkText(p: DPaper): string {
  const c = p.claim
  const proto = p.protocol ?? c?.protocol
  const ds = p.dataset ?? c?.matched_task ?? '–'
  if (proto === 'unklar' || !c?.matched_task) return `${ds} (split not stated)`
  if (c.matched_task && ETT_TASKS.includes(c.matched_task)) return `${c.matched_task} · horizon ${c.horizon ?? 96}`
  return TASK_TEXT[c.matched_task] ?? c.matched_task
}

// Runtime shown while a run is going: the last event's runtime plus the wall time since that event (times `rate` for a
// sped-up replay), so the timer ticks between events. Never below the event runtime; frozen once the run is not running.
export function tickedRuntime(eventRuntime: number, sinceEventMs: number, rate: number, running: boolean): number {
  const base = Number.isFinite(eventRuntime) ? Math.max(0, eventRuntime) : 0
  if (!running) return base
  return base + Math.max(0, sinceEventMs) / 1000 * rate
}

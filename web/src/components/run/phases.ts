import type { Event } from '../../data'
import type { Tone } from '../../labels'
import { VERDICT_SENTENCE } from '../../labels'

// Turns the flat event list of a run into phases with grouped rows. Pure: every number comes from an event, a missing value drops the line.

export type Actor = 'tavily' | 'nemotron' | 'sandbox' | 'harness' | 'system'
export type PhaseId = 'claim' | 'setup' | 'plan' | 'code' | 'sandbox' | 'seeds' | 'verdict' | 'community'
export type Expand =
  | { kind: 'plan'; text: string }
  | { kind: 'code'; hash: string; version: number; inline?: string }
  | { kind: 'setup'; items: { topic: string; value: string; source: string; quote: string }[] }
  | { kind: 'quote'; quote: string }
  | { kind: 'error'; text: string }

export type Row = {
  id: string
  ts: number
  actor: Actor
  role?: 'fast' | 'code' | 'reasoning'
  model?: string
  tavily?: 'live' | 'stored'   // actor tavily only: fetched live in this run or loaded from a stored copy
  title: string
  detail?: string
  tone: Tone
  expand?: Expand
  raw: Event[]
}

export type PhaseState = 'pending' | 'active' | 'done'
export type Phase = {
  id: PhaseId
  title: string
  short: string
  actors: Actor[]
  what: string
  why: string
  state: PhaseState
  summary: string | null
  rows: Row[]
}

export type Attempt = { seed: number; hash: string; measured: number; seconds: number | null }
export type Stats = { mean: number; std: number; n: number }

export type RunModel = {
  title: string | null
  claimText: string | null
  metric: string | null
  target: number | null
  tolerance: number | null
  seedsTotal: number | null
  models: Partial<Record<'fast' | 'code' | 'reasoning', string>>
  phases: Phase[]
  attempts: Attempt[]          // valid measurements of the code the seeds ran with (discarded graph-repair codes removed)
  stats: Stats | null
  tavilyCalls: number
  tavilyKind: 'live' | 'stored' | null
  warnings: string[]
  verdict: Event | null
  startTs: number | null
  endTs: number | null
  costNet: number | null
  graphTask: boolean
  waiting: string | null       // what the loop is doing right now, for the pulsing line
  firstMeasurement: boolean
  llmCalls: number             // live only (llm_call events)
  llmCached: number
}

export const ACTOR_LABEL: Record<Actor, string> = { tavily: 'Tavily', nemotron: 'Nemotron', sandbox: 'Sandbox', harness: 'Harness', system: 'System' }

const SHORT: Record<PhaseId, string> = { claim: 'Claim', setup: 'Setup', plan: 'Plan', code: 'Code', sandbox: 'Sandbox', seeds: 'Seeds', verdict: 'Verdict', community: 'Community' }
const PHASE_TEXT: Record<PhaseId, { title: string; actors: Actor[]; what: string; why: string }> = {
  claim: {
    title: 'Check the claimed number', actors: ['tavily', 'nemotron'],
    what: 'Tavily reads the paper again and Nemotron finds the reported number in its results table.',
    why: 'The claim came from discovery. A second, independent reading protects against misreads.',
  },
  setup: {
    title: 'Collect setup details', actors: ['tavily', 'nemotron'],
    what: 'Tavily fetches the code repository README. Nemotron extracts hyperparameters and data splits, each with a quote.',
    why: 'The agent should reuse what the authors state instead of guessing. Anything without a verbatim quote is dropped.',
  },
  plan: {
    title: 'Plan the experiment', actors: ['nemotron'],
    what: 'Nemotron (reasoning model) writes a short reproduction plan from the paper details.',
    why: 'A written plan makes the experiment checkable and keeps the code close to the paper method.',
  },
  code: {
    title: 'Write the code', actors: ['nemotron'],
    what: 'Nemotron (code model) writes a training script that outputs predictions only.',
    why: 'The agent never sees the test labels and never reports a score itself.',
  },
  sandbox: {
    title: 'Run in the sandbox', actors: ['sandbox', 'harness'],
    what: 'Seed 0 trains in an isolated sandbox. The harness scores it. For graph models a graph check follows: all graph edges are removed and the same code runs again.',
    why: 'If accuracy does not drop without the graph, the code ignores the graph and cannot be the paper method.',
  },
  seeds: {
    title: 'Repeat with more seeds', actors: ['sandbox', 'harness'],
    what: 'The remaining seeds run in parallel, each in its own sandbox branch with a different random seed.',
    why: 'One run can be lucky. The verdict uses the mean over all seeds and reports the spread.',
  },
  verdict: {
    title: 'Verdict', actors: ['harness'],
    what: 'The harness compares the mean with the claimed value inside a fixed tolerance.',
    why: 'The tolerance is set before the run, so the verdict cannot be tuned afterwards.',
  },
  community: {
    title: 'What others found', actors: ['tavily', 'nemotron'],
    what: 'After the verdict, Tavily looks for public discussion of the paper (GitHub issues, reviews). Nemotron lists what it says, each statement with a verbatim quote.',
    why: 'Words next to measurement. These notes are reported only and never change the verdict.',
  },
}
const ORDER: PhaseId[] = ['claim', 'setup', 'plan', 'code', 'sandbox', 'seeds', 'verdict', 'community']

const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : x != null && x !== '' && Number.isFinite(Number(x)) ? Number(x) : null)
const short = (s: string, n = 140) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
const lastLine = (s: unknown) => String(s ?? '').split('\n').map((x) => x.trim()).filter(Boolean).pop() ?? ''
const host = (url: string) => { try { const u = new URL(url); return u.host + u.pathname.replace(/\/$/, '') } catch { return url } }

export const fmtSeconds = (s: number) => (s < 90 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`)
export const fmtClock = (s: number) => `${String(Math.floor(Math.max(0, s) / 60)).padStart(2, '0')}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, '0')}`

export function fmtValue(metric: string | null, v: number | null, digits?: number): string {
  if (v == null) return '–'
  return metric === 'accuracy' ? `${(v * 100).toFixed(digits ?? 1)}%` : v.toFixed(digits ?? 3)
}

export function median(xs: number[]): number | null {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2
}

export function buildRunModel(events: Event[], opts: { running: boolean; metricHint?: string | null }): RunModel {
  const runStart = events.find((e) => e.type === 'run_start')
  const claimStart = events.find((e) => e.type === 'claim_start')
  const models = (runStart?.models ?? {}) as RunModel['models']
  const metric = (claimStart?.metric as string | undefined) ?? opts.metricHint ?? null   // the hint covers the first phase, before claim_start arrives
  const target = num(claimStart?.target)
  const seedsTotal = num(runStart?.seeds)
  const modelName = (role: 'fast' | 'code' | 'reasoning') => (models[role] ? String(models[role]).replace(/^nvidia\//i, '') : undefined)
  const discarded = new Set(events.filter((e) => (e.type === 'graph_repair' || e.type === 'fault_repair') && e.result === 'repariert').map((e) => String(e.from_hash)))
  const graphTask = events.some((e) => e.type === 'ablation') || metric === 'accuracy'

  const rows: Record<PhaseId, Row[]> = { claim: [], setup: [], plan: [], code: [], sandbox: [], seeds: [], verdict: [], community: [] }
  const versions: string[] = []
  const codeInline = new Map<string, string>()
  const planFull = events.find((e) => e.type === 'plan_text')?.text as string | undefined
  const seen = new Set(events.map((e) => e.type))
  let tavilyCalls = 0
  const warnings: string[] = []
  const tavilyRows = new Map<string, Row>()   // url -> row, so the source event can add what was read
  const versionOf = (h: string) => { if (!versions.includes(h)) versions.push(h); return versions.indexOf(h) + 1 }

  events.forEach((e, i) => {
    const id = String(i)
    const push = (phase: PhaseId, r: Omit<Row, 'id' | 'ts' | 'raw'>) => rows[phase].push({ id, ts: e.ts, raw: [e], ...r })
    switch (e.type) {
      case 'warning': warnings.push(String(e.text ?? '')); break
      case 'code_text': codeInline.set(String(e.code_hash), String(e.code ?? '')); break
      case 'tavily_call': {
        tavilyCalls++
        const url = String(e.target ?? '')
        const role = Number(e.role)
        const phase: PhaseId = role === 4 ? 'setup' : role === 3 || role === 6 ? 'community' : 'claim'
        const isUrl = /^https?:/.test(url)
        const r: Row = {
          id, ts: e.ts, raw: [e], actor: 'tavily', tone: 'amber',
          title: role === 4 ? 'Tavily Extract: code repository README' : role === 2 ? 'Tavily Extract: paper page' : role === 6 ? 'Tavily Search: errata and newer versions' : role === 3 ? (e.endpoint === 'crawl' ? 'Tavily Crawl: repository issues' : e.endpoint === 'extract' ? 'Tavily Extract: discussion pages' : 'Tavily Search: public discussion') : isUrl ? 'Tavily Extract' : 'Tavily Search',
          detail: isUrl ? host(url) : short(url, 120),
        }
        rows[phase].push(r)
        if (isUrl) tavilyRows.set(url, r)
        break
      }
      case 'source': {
        const url = String(e.url ?? '')
        const existing = tavilyRows.get(url)
        const chars = num(e.n_chars)
        const read = chars != null ? `${chars.toLocaleString('en-US')} characters read` : ''
        if (existing) {
          existing.detail = [existing.detail, read].filter(Boolean).join(' · ')
          if (typeof e.mode === 'string') existing.tavily = e.mode === 'live' ? 'live' : 'stored'
          existing.raw.push(e)
        } else if (e.kind === 'community') {
          push('community', { actor: 'tavily', tavily: e.mode === 'live' ? 'live' : 'stored', tone: 'amber', title: e.mode === 'live' ? 'Tavily read a discussion page' : 'Discussion page loaded from the 7-day cache', detail: [host(url), read].filter(Boolean).join(' · ') })
        } else {
          push(e.kind === 'readme' ? 'setup' : 'claim', {
            actor: 'tavily', tavily: e.mode === 'live' ? 'live' : 'stored', tone: 'amber', title: e.mode === 'live' ? `Tavily Extract: ${e.kind === 'readme' ? 'README' : 'paper page'}` : `Saved copy of the ${e.kind === 'readme' ? 'README' : 'paper page'} loaded`,
            detail: [host(url), read].filter(Boolean).join(' · '),
          })
        }
        break
      }
      case 'target_check': {
        const agrees = e.agrees === true
        push('claim', {
          actor: 'nemotron', role: 'fast', model: modelName('fast'), tone: agrees ? 'green' : 'amber',
          title: agrees ? 'Second reading of the paper confirms the number' : 'Second reading of the paper differs from discovery',
          detail: `Paper: ${fmtValue(metric, num(e.verified_value), 2)} · Discovery: ${fmtValue(metric, num(e.configured_value), 2)}${e.from_store ? ' · taken from a stored extraction of this paper, no new model call' : ''}`,
        })
        break
      }
      case 'infrastructure_error': {
        push('sandbox', { actor: 'system', tone: 'red', title: 'The sandbox setup failed (infrastructure, not a verdict)', detail: short(String(e.message ?? '').split('\n')[0], 160) })
        break
      }
      case 'paper_summary': {
        const t = typeof e.text === 'string' ? e.text : ''
        const n = Array.isArray(e.institutions) ? e.institutions.length : 0
        push('claim', {
          actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'violet', title: 'Plain-language summary of the paper written',
          detail: [t ? 'Two or three simple sentences from the arXiv abstract.' : null, n ? `${n} affiliation${n === 1 ? '' : 's'} read from the paper text, each checked against a quote.` : null, 'Reported only, never part of the check.'].filter(Boolean).join(' '),
        })
        break
      }
      case 'target_resolved': {
        const v = num(e.value)
        push('claim', {
          actor: 'nemotron', role: 'fast', model: modelName('fast'), tone: 'green',
          title: `Target ${fmtValue(metric, v, 2)} found in ${e.table || 'the paper'}`,
          detail: `${e.column_check === 'ok' ? 'Table column verified: the number belongs to the proposed method.' : 'Table column could not be verified.'}${e.from_store ? ' Number read from a stored extraction of this paper (no new model call).' : ''}`,
          expand: e.quote ? { kind: 'quote', quote: String(e.quote) } : undefined,
        })
        break
      }
      case 'setup_details': {
        const items = ((e.details ?? []) as Record<string, unknown>[]).map((d) => ({ topic: String(d.topic ?? ''), value: String(d.value ?? ''), source: String(d.source ?? 'paper').split('@')[0], quote: String(d.quote ?? '') }))
        push('setup', {
          actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'amber',
          title: `${e.n ?? items.length} setup details confirmed by quotes`,
          detail: num(e.dropped) ? `${e.dropped} dropped because no matching quote was found.` : undefined,
          expand: items.length ? { kind: 'setup', items } : undefined,
        })
        break
      }
      case 'plan': {
        push('plan', { actor: 'nemotron', role: 'reasoning', model: modelName('reasoning'), tone: 'violet', title: 'Experiment plan written', expand: { kind: 'plan', text: String(planFull ?? e.text ?? '') } })
        break
      }
      case 'code_written': {
        const h = String(e.code_hash)
        const v = versionOf(h)
        push('code', { actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'violet', title: `Code version ${v} written`, expand: { kind: 'code', hash: h, version: v } })
        break
      }
      case 'repair': {
        const h = String(e.from_hash)
        versionOf(h)
        push('sandbox', {
          actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'red',
          title: 'The code crashed in the sandbox, Nemotron rewrites it',
          detail: short(lastLine(e.error), 160) || undefined,
          expand: e.error ? { kind: 'error', text: String(e.error) } : undefined,
        })
        break
      }
      case 'target_blocked': {
        push('claim', { actor: 'system', tone: 'amber', title: 'Claim verification stopped',
          detail: String(e.reason ?? 'The target could not be verified. No experiment was started.') })
        break
      }
      case 'repo_diff': {
        if (e.result === 'incomplete' || e.result === 'no_deviation' && (Number(e.dropped) > 0 || Array.isArray(e.truncation) && e.truncation.length > 0)) push('code', { actor: 'system', tone: 'amber', title: 'Repository comparison incomplete',
          detail: 'Files were truncated or findings could not be verified. Method fidelity remains unchecked.' })
        break
      }
      case 'attempt': {
        if (e.method_incomplete) {
          push('code', { actor: 'system', tone: 'amber', title: 'Incomplete method: code rejected before sandbox execution',
            expand: e.error ? { kind: 'error', text: String(e.error) } : undefined })
          break
        }
        const seed = num(e.seed) ?? 0
        const h = String(e.code_hash ?? '')
        if (h) versionOf(h)
        const m = num(e.measured)
        const secs = num(e.seconds)
        const phase: PhaseId = seed === 0 ? 'sandbox' : 'seeds'
        const attempt = num(e.n) ?? 1
        const name = `Seed ${seed}${attempt > 1 ? `, attempt ${attempt}` : ''}`
        if (m == null) {
          push(phase, { actor: 'sandbox', tone: 'red', title: `${name} did not produce a valid run`, detail: short(lastLine(e.error), 160) || undefined, expand: e.error ? { kind: 'error', text: String(e.error) } : undefined })
        } else {
          push(phase, {
            actor: 'harness', tone: 'gray', title: `${name} finished, harness scored ${fmtValue(metric, m)}`,
            detail: secs != null ? `Trained in its own sandbox branch in ${fmtSeconds(secs)}. The agent only wrote predictions.` : 'The agent only wrote predictions.',
          })
        }
        break
      }
      case 'infra_retry': {
        const seed = num(e.seed) ?? 0
        push(seed === 0 ? 'sandbox' : 'seeds', { actor: 'system', tone: 'gray', title: `Sandbox service timed out on seed ${seed}, trying again`, detail: 'The same code and seed run again in a new branch.' })
        break
      }
      case 'ablation': {
        const v = String(e.verdict ?? '')
        const full = num(e.accuracy_full), abl = num(e.accuracy_ablated)
        const round = num(e.runde) ?? 0
        const after = round > 0 ? ` after repair ${round}` : ''
        const nums = full != null && abl != null ? `${fmtValue('accuracy', full)} with graph edges, ${fmtValue('accuracy', abl)} with edges removed` : undefined
        if (v === 'Graph genutzt') push('sandbox', { actor: 'harness', tone: 'green', title: `Graph check passed${after}`, detail: nums })
        else if (v === 'Graph nicht genutzt') push('sandbox', { actor: 'harness', tone: 'red', title: `Graph check failed${after}: the code ignores the graph`, detail: nums })
        else if (v === 'unklar') push('sandbox', { actor: 'harness', tone: 'amber', title: `Graph check inconclusive${after}`, detail: nums })
        else push('sandbox', { actor: 'harness', tone: 'gray', title: `Graph check could not be measured${after}` })
        break
      }
      case 'plan_retry': {
        push('plan', { actor: 'nemotron', role: 'reasoning', model: modelName('reasoning'), tone: 'amber', title: `The first answer was ${e.reason === 'too_long' ? 'far too long' : 'code, not a plan'}: Nemotron is asked again for a short plan` })
        break
      }
      case 'fault_repair': {
        const r = String(e.result)
        const base = Object.entries((e.baselines ?? {}) as Record<string, unknown>).filter(([, v]) => num(v) != null).map(([k, v]) => `${k} ${fmtValue(metric, num(v))}`).join(', ')
        push('sandbox', {
          actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'violet',
          title: `Result worse than a trivial baseline, round ${e.runde}: Nemotron gets the numbers and fixes the code`,
          detail: r === 'repariert'
            ? `The agent's code is likely wrong, not the paper. It measured ${fmtValue(metric, num(e.measured))}, trivial baselines: ${base || 'n/a'}. The fixed code is checked again.`
            : 'The rewritten code did not run, the previous code is kept.',
        })
        break
      }
      case 'graph_repair': {
        const r = String(e.result)
        push('sandbox', {
          actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'violet',
          title: `Graph repair, round ${e.runde}: Nemotron rewrites the code`,
          detail: r === 'repariert' ? 'The first code ignored the graph. New code version is checked again.' : 'The rewritten code did not run, the previous code is kept.',
        })
        break
      }
      case 'community': {
        const n = num(e.n_findings) ?? 0
        push('community', { actor: 'nemotron', role: 'fast', model: modelName('fast'), tone: n ? 'violet' : 'gray',
          title: n ? `${n} public note${n === 1 ? '' : 's'} found` : 'No public discussion found',
          detail: `${e.mode === 'cache' ? 'From the 7-day cache. ' : ''}Each note carries a verbatim quote; ${num(e.dropped) ?? 0} statement${num(e.dropped) === 1 ? '' : 's'} without a quote in the page dropped. Not part of the verdict.` })
        break
      }
      case 'objection_classified': {
        const t = String(e.otype ?? '')
        push('community', { actor: 'nemotron', role: 'fast', model: modelName('fast'), tone: t === 'not_measurable' ? 'gray' : 'violet',
          title: t === 'not_measurable' ? 'Objection kept as text' : `Objection turned into a test: ${t.replace(/_/g, ' ')}`,
          detail: e.note ? String(e.note) : undefined, expand: e.quote ? { kind: 'quote', quote: String(e.quote) } : undefined })
        break
      }
      case 'objection_experiment': {
        if (e.stage === 'seed') push('community', { actor: 'sandbox', tone: e.measured == null ? 'red' : 'green', title: `Objection test, seed ${e.seed}`,
          detail: e.measured == null ? 'No valid measurement.' : `${fmtValue(metric, num(e.measured))} measured by the harness in its own branch.` })
        break
      }
      case 'objection_checked': {
        const lab: Record<string, string> = { bestaetigt: 'confirmed', widerlegt: 'refuted', offen: 'open', 'nicht pruefbar': 'not measurable' }
        push('community', { actor: 'harness', tone: e.status === 'widerlegt' ? 'green' : e.status === 'bestaetigt' ? 'amber' : 'gray',
          title: `Objection ${lab[String(e.status)] ?? e.status}`, detail: 'Tested by us with a rule fixed in advance. Not part of the verdict.', expand: e.quote ? { kind: 'quote', quote: String(e.quote) } : undefined })
        break
      }
      case 'explanations': {
        const n = num(e.n_items) ?? 0
        push('community', { actor: 'nemotron', role: 'fast', model: modelName('fast'), tone: n ? 'violet' : 'gray',
          title: n ? `${n} possible explanation${n === 1 ? '' : 's'} (unverified)` : 'No explanation found in public sources', detail: 'Hypotheses with a quote, not tested by us.' })
        break
      }
      case 'verdict': {
        const s = VERDICT_SENTENCE[String(e.urteil)] ?? String(e.urteil)
        push('verdict', { actor: 'harness', tone: e.urteil === 'reproduziert' ? 'green' : e.urteil === 'teilweise' ? 'amber' : e.urteil === 'nicht reproduziert' ? 'red' : 'gray', title: s, detail: `Mean ${fmtValue(metric, num(e.mean))} over the seeds, spread ${fmtValue(metric, num(e.std))}.` })
        break
      }
    }
  })

  // valid measurements of the code the seeds ran with; last value per seed
  const bySeed = new Map<number, Attempt>()
  for (const e of events) {
    if (e.type !== 'attempt') continue
    const m = num(e.measured)
    if (m == null || discarded.has(String(e.code_hash))) continue
    bySeed.set(num(e.seed) ?? 0, { seed: num(e.seed) ?? 0, hash: String(e.code_hash), measured: m, seconds: num(e.seconds) })
  }
  const attempts = [...bySeed.values()].sort((a, b) => a.seed - b.seed)
  let stats: Stats | null = null
  if (attempts.length) {
    const xs = attempts.map((a) => a.measured)
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length
    stats = { mean, std: xs.length > 1 ? Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (xs.length - 1)) : 0, n: xs.length }
  }

  // rows for code versions found only through attempts/repairs (version >= 2) are shown inside the sandbox phase
  versions.forEach((h, idx) => {
    if (idx === 0 || rows.code.some((r) => r.expand?.kind === 'code' && r.expand.hash === h)) return
    const firstAttempt = rows.sandbox.find((r) => r.raw[0].code_hash === h)
    rows.sandbox.push({ id: `v${idx}`, ts: firstAttempt?.ts ?? 0, raw: [], actor: 'nemotron', role: 'code', model: modelName('code'), tone: 'violet', title: `Code version ${idx + 1}`, expand: { kind: 'code', hash: h, version: idx + 1, inline: codeInline.get(h) } })
  })
  for (const r of rows.code) if (r.expand?.kind === 'code') r.expand.inline = codeInline.get(r.expand.hash)
  rows.sandbox.sort((a, b) => a.ts - b.ts)

  const okSeeds = new Set(attempts.map((a) => a.seed))
  const seed0 = okSeeds.has(0)
  const hasAblation = seen.has('ablation')
  const laterSeeds = events.some((e) => e.type === 'attempt' && (num(e.seed) ?? 0) > 0)
  const verdictSeen = seen.has('verdict')
  const done: Record<PhaseId, boolean> = {
    claim: seen.has('target_resolved') || seen.has('setup_details') || seen.has('claim_start') || seen.has('plan'),
    setup: seen.has('setup_details') || seen.has('claim_start') || seen.has('plan'),   // claim_start comes after the setup phase
    plan: seen.has('plan') && (seen.has('code_written') || seen.has('attempt')),   // stays open while the next step starts: the plan can be read live
    code: seen.has('code_written') && (seen.has('attempt') || seen.has('ablation')),
    sandbox: laterSeeds || verdictSeen || (seedsTotal === 1 && seed0 && (hasAblation || !graphTask)),
    seeds: verdictSeen,
    verdict: verdictSeen,
    community: seen.has('community'),
  }
  // The community check runs after the verdict and only in live product runs: the phase exists once its first event arrives.
  const order = ORDER.filter((k) => k !== 'community' || rows.community.length > 0 || seen.has('community'))
  if (!opts.running) for (const k of order) if (rows[k].length && !done[k]) done[k] = verdictSeen   // aborted run: nothing stays "active"
  const activeId = opts.running ? order.find((k) => !done[k]) ?? null : null

  const first = events[0]?.ts ?? null
  const last = events.length ? events[events.length - 1].ts : null
  const planText = planFull ?? (events.find((e) => e.type === 'plan')?.text as string | undefined) ?? ''
  const planSteps = planText.split('\n').filter((l) => /^\s*\d+[.)]/.test(l)).length
  const target2 = events.find((e) => e.type === 'target_resolved')
  const setupEv = events.find((e) => e.type === 'setup_details')
  const verdictEv = events.filter((e) => e.type === 'verdict').pop() ?? null
  const ab = events.filter((e) => e.type === 'ablation').pop()
  const repairs = events.filter((e) => e.type === 'graph_repair').length
  const faultRepairs = events.filter((e) => e.type === 'fault_repair').length
  const summaries: Record<PhaseId, string | null> = {
    claim: target2 ? `Target ${fmtValue(metric, num(target2.value), 2)} confirmed in ${target2.table || 'the paper'}` : null,
    setup: setupEv ? `${setupEv.n} setup details confirmed by quotes${num(setupEv.dropped) ? `, ${setupEv.dropped} dropped` : ''}` : done.setup ? 'No setup details used' : null,
    plan: seen.has('plan') ? `Plan written${planSteps ? `, ${planSteps} steps` : ''}` : null,
    code: versions.length ? `Code version 1${codeInline.get(versions[0]) ? `, ${codeInline.get(versions[0])!.split('\n').length} lines` : ''}${versions.length > 1 ? `, ${versions.length - 1} rewrite${versions.length > 2 ? 's' : ''} later` : ''}` : null,
    sandbox: [
      seed0 ? `Seed 0 scored ${fmtValue(metric, attempts.find((a) => a.seed === 0)!.measured)}` : null,
      ab ? (ab.verdict === 'Graph genutzt' ? 'graph check passed' : ab.verdict === 'Graph nicht genutzt' ? 'graph check failed' : 'graph check inconclusive') : null,
      repairs ? `${repairs} graph repair${repairs > 1 ? 's' : ''}` : null,
      faultRepairs ? `${faultRepairs} code fix${faultRepairs > 1 ? 'es' : ''} after a result worse than a baseline` : null,
    ].filter(Boolean).join(' · ') || null,
    seeds: stats && seedsTotal ? `${stats.n} of ${seedsTotal} seeds, mean ${fmtValue(metric, stats.mean)} ± ${fmtValue(metric, stats.std)}` : null,
    verdict: verdictEv ? (VERDICT_SENTENCE[String(verdictEv.urteil)] ?? String(verdictEv.urteil)) : null,
    community: (() => { const c = events.filter((e) => e.type === 'community').pop(); return c ? (num(c.n_findings) ? `${c.n_findings} public note${c.n_findings === 1 ? '' : 's'}, not part of the verdict` : 'No public discussion found') : null })(),
  }

  const medSeconds = median(attempts.filter((a) => a.seed > 0 && a.seconds != null).map((a) => a.seconds as number))
  let waiting: string | null = null
  if (activeId === 'claim') waiting = 'Tavily is reading the paper and Nemotron is finding the reported number.'
  else if (activeId === 'setup') waiting = 'Collecting setup details from the paper and the repository.'
  else if (activeId === 'plan') waiting = seen.has('plan') ? 'The plan is written. Nemotron is now writing the code.' : 'Nemotron is writing the experiment plan.'
  else if (activeId === 'code') waiting = seen.has('code_written') ? 'The code is written. The sandbox runs a static check and a quick smoke test before training.' : 'Nemotron is writing the training code.'
  else if (activeId === 'sandbox') {
    const lastAb = events.filter((e) => e.type === 'ablation').pop()
    const rewriting = !!lastAb && lastAb.verdict === 'Graph nicht genutzt' && repairs < 2 && repairs + faultRepairs < 3
    waiting = !seed0 ? 'Training seed 0 in the sandbox. This can take several minutes.'
      : graphTask && !hasAblation ? 'Graph check: running the same code again without graph edges.'
        : rewriting ? 'The graph check failed. Nemotron is rewriting the code.'
          : !graphTask && seedsTotal !== 1 ? 'Checking the result against trivial baselines.'
            : 'Starting the remaining seeds in parallel.'
  }
  else if (activeId === 'seeds') {
    const left = seedsTotal ? Math.max(0, seedsTotal - okSeeds.size) : null
    waiting = left === 0 ? 'All seeds finished. The harness is computing the verdict.' : `${left != null ? `${left} seed${left === 1 ? '' : 's'} still training` : 'Seeds are training'} in parallel sandbox branches${medSeconds ? `, about ${fmtSeconds(medSeconds)} each` : ''}.`
  } else if (activeId === 'verdict') waiting = 'The harness is computing the verdict.'
  else if (activeId === 'community') waiting = 'The verdict is final. Tavily is checking public discussion of the paper; this never changes the verdict.'

  const phases: Phase[] = order.map((id) => {
    const t = PHASE_TEXT[id]
    const state: PhaseState = done[id] ? 'done' : activeId === id ? 'active' : 'pending'
    const what = id === 'sandbox' && !graphTask ? 'Seed 0 trains in an isolated sandbox and the harness scores it.' : t.what
    const why = id === 'sandbox' && !graphTask ? 'A first run catches crashes before time goes into more seeds.' : t.why
    return { id, title: t.title, short: SHORT[id], actors: t.actors, what, why, state, summary: summaries[id], rows: rows[id] }
  })

  // The tavily_call event carries no mode; the source events do. A run that read any source live counts as live, otherwise stored.
  const srcModes = events.filter((e) => e.type === 'source' && typeof e.mode === 'string').map((e) => e.mode)
  const tavilyKind: 'live' | 'stored' | null = srcModes.length ? (srcModes.includes('live') ? 'live' : 'stored') : null
  if (tavilyKind) for (const ph of Object.values(rows)) for (const r of ph) if (r.actor === 'tavily' && !r.tavily) r.tavily = tavilyKind

  const cost = [...events].reverse().find((e) => e.type === 'llm_call')
  const end = events.find((e) => e.type === 'run_end')

  return {
    title: (runStart?.title as string | undefined) ?? null, claimText: (claimStart?.text as string | undefined) ?? null, metric, target,
    tolerance: num(claimStart?.tolerance), seedsTotal, models, phases, attempts, stats, tavilyCalls, tavilyKind, warnings, verdict: verdictEv,
    startTs: first, endTs: last, costNet: num(end?.cost_net_usd) ?? num(cost?.total_net_usd), graphTask, waiting, firstMeasurement: attempts.length > 0,
    llmCalls: events.filter((e) => e.type === 'llm_call').length, llmCached: events.filter((e) => e.type === 'llm_call' && e.cached === true).length,
  }
}

// "Plan step -> implemented in: X" comment block at the top of the agent's code (the agent's own statement, not checked by us).
export function planMapping(code: string): string[] | null {
  const lines = code.split('\n').slice(0, 80)
  const at = lines.findIndex((l) => /^\s*#.*plan step/i.test(l))
  if (at < 0) return null
  let from = at
  while (from > 0 && /^\s*#/.test(lines[from - 1])) from--
  let to = at
  while (to + 1 < lines.length && /^\s*#/.test(lines[to + 1])) to++
  const block = lines.slice(from, to + 1).map((l) => l.replace(/^\s*#+\s?/, '').trimEnd()).filter((l) => l.trim() && !/^[-=_*\s]+$/.test(l))
  return block.length ? block.slice(0, 25) : null
}

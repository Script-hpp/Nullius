// "Watch the agent fix its own code": the repair loop of one claim, told as cards in order. Pure functions over the run's events
// (events.jsonl) and the stored code versions (code_<hash>.py). Facts only: nothing here is written by a model after the fact.
import type { Event } from './data'

export type Outcome = { tone: 'ok' | 'fail' | 'unknown'; text: string }
export type Card =
  | { kind: 'candidates'; text: string }
  | { kind: 'repo_diff'; text: string; fromHash: string | null; toHash: string | null }
  | {
      kind: 'repair'; source: 'error' | 'fault' | 'graph'; fromVersion: number; toVersion: number | null; fromHash: string; toHash: string | null
      problemTitle: string; problem: string; search: { query: string; links: { url: string; domain: string }[] } | null; outcome: Outcome
    }

const str = (x: unknown) => (typeof x === 'string' ? x : '')
const num = (x: unknown) => (typeof x === 'number' && Number.isFinite(x) ? x : null)

export const domainOf = (url: string) => { try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url } }

// Last meaningful line of an error text, German labels of the loop translated.
export function problemOf(error: string): { title: string; line: string } {
  const lines = error.split('\n').map((l) => l.trim()).filter(Boolean)
  const last = (lines[lines.length - 1] ?? '').slice(0, 240)
  if (/^Statische Pr/.test(error)) return { title: 'The static code check rejected the code', line: error.replace(/^Statische Pr[^:]*:\s*/, '').split('\n')[0].replace('URL im Code', 'URL in code').replace('Zeile', 'line') }
  if (/^Smoke run \(SMOKE=1\) crashed/.test(error)) return { title: 'The quick form test crashed', line: last }
  if (/^Smoke run \(SMOKE=1\) output has the wrong form/.test(error)) return { title: 'The quick form test found the wrong output form', line: lines[0].replace(/^Smoke run \(SMOKE=1\) output has the wrong form:\s*/, '').slice(0, 240) }
  return { title: 'The code crashed', line: last || 'no error text recorded' }
}

const fmt = (x: number | null) => (x == null ? 'n/a' : Math.abs(x) < 1.5 ? x.toFixed(3) : x.toFixed(2))

function outcomeOf(events: Event[], claimId: string, hash: string | null): Outcome {
  if (!hash) return { tone: 'unknown', text: 'No new version was recorded after this fix.' }
  const a = events.find((e) => e.type === 'attempt' && e.claim_id === claimId && e.code_hash === hash)
  if (!a) return { tone: 'unknown', text: 'No result was recorded for the new version.' }
  const status = str(a.status)
  if ((status === 'stabil' || status === 'abweichung') && num(a.measured) != null) return { tone: 'ok', text: `It ran. Valid measurement: ${fmt(num(a.measured))}.` }
  if (status === 'timeout') return { tone: 'fail', text: 'It ran too long and was stopped.' }
  const p = problemOf(str(a.error))
  return { tone: 'fail', text: `${status === 'ungueltig' ? 'It was rejected' : 'It failed again'}: ${p.line}` }
}

// Version numbers: a code that is nobody's repair result is version 1; a repair result is its source plus 1.
function versions(pairs: [string, string | null][]): Map<string, number> {
  const v = new Map<string, number>()
  for (const [, to] of pairs) if (to) v.set(to, 0)
  for (const [from] of pairs) if (!v.has(from)) v.set(from, 1)
  for (let pass = 0; pass < pairs.length + 1; pass++) for (const [from, to] of pairs) if (to && v.get(from)) v.set(to, (v.get(from) as number) + 1)
  return v
}

export function buildStory(events: Event[], claimId: string): Card[] {
  const idx = events.map((e, i) => [e, i] as const).filter(([e]) => e.claim_id === claimId || e.claim_id === undefined)
  // Error repairs: the repair event only knows the failed code. Its result is the next unclaimed attempt of the same seed with n + 1
  // (parallel candidates interleave, so first come, first served). Fault and graph repairs name both hashes themselves.
  const claimed = new Set<number>()
  const resolved = new Map<number, string | null>()
  for (const [e, i] of idx) {
    if (e.type !== 'repair') continue
    const failed = [...idx].reverse().find(([x, j]) => j < i && x.type === 'attempt' && x.code_hash === e.from_hash)
    const n = failed ? num(failed[0].n) : null
    const next = n == null ? undefined : idx.find(([x, j]) => j > i && !claimed.has(j) && x.type === 'attempt' && x.seed === e.seed && num(x.n) === n + 1)
    if (next) claimed.add(next[1])
    resolved.set(i, next ? str(next[0].code_hash) : null)
  }
  const pairs: [string, string | null][] = []
  for (const [e, i] of idx) {
    if (e.type === 'repair') pairs.push([str(e.from_hash), resolved.get(i) ?? null])
    else if ((e.type === 'fault_repair' || e.type === 'graph_repair') && e.result === 'repariert') pairs.push([str(e.from_hash), str(e.to_hash)])
  }
  const ver = versions(pairs)
  const cards: Card[] = []
  let lastRepair = -1
  for (const [e, i] of idx) {
    if (e.type === 'candidates') {
      const k = num(e.k) ?? 0
      cards.push({ kind: 'candidates', text: e.reason === 'val'
        ? `${k} versions of the code were written; the one with the best score on validation data was kept.`
        : `${k} versions of the code were written; none could be scored on validation data, so the first one that ran was kept.` })
    } else if (e.type === 'repo_diff') {
      const r = str(e.result)
      const devs = Array.isArray(e.deviations) ? e.deviations.length : 0
      const text = r === 'no_deviation' ? 'We compared the code with the files of the paper’s repository and found no deviation we could confirm.'
        : r === 'adopted' ? `We compared the code with the paper’s repository. ${devs} confirmed deviation${devs === 1 ? '' : 's'} led to a rewrite, and the new version passed all checks and was adopted.`
        : r === 'kept' ? `We compared the code with the paper’s repository and tried a rewrite for ${devs} deviation${devs === 1 ? '' : 's'}. The new version was not adopted; the previous code was kept.`
        : 'The comparison with the paper’s repository could not be completed; the code was left as it was.'
      cards.push({ kind: 'repo_diff', text, fromHash: str(e.from_hash) || null, toHash: str(e.to_hash) || null })
    } else if (e.type === 'repair' || ((e.type === 'fault_repair' || e.type === 'graph_repair') && e.result === 'repariert')) {
      const from = str(e.from_hash)
      const to = e.type === 'repair' ? resolved.get(i) ?? null : str(e.to_hash)
      let source: 'error' | 'fault' | 'graph' = 'error'
      let title = ''
      let problem = ''
      let search: { query: string; links: { url: string; domain: string }[] } | null = null
      if (e.type === 'repair') {
        const p = problemOf(str(e.error))
        title = p.title; problem = p.line
        const s = [...idx].reverse().find(([x, j]) => j < i && j > lastRepair && x.type === 'error_search')
        if (s) {
          const urls = (Array.isArray(s[0].urls) ? s[0].urls : []).map(str).filter(Boolean)
          search = { query: str(s[0].query), links: urls.map((url) => ({ url, domain: domainOf(url) })) }
        }
        lastRepair = i
      } else if (e.type === 'fault_repair') {
        source = 'fault'
        const base = Object.entries((e.baselines ?? {}) as Record<string, unknown>).filter(([, v]) => num(v) != null).map(([k, v]) => `${k} ${fmt(num(v))}`).join(', ')
        title = 'The result was no better than a trivial baseline'
        problem = `It measured ${fmt(num(e.measured))}${base ? `, trivial baselines: ${base}` : ''}. The code is likely wrong, not the paper.`
      } else {
        source = 'graph'
        title = 'The graph check showed the code ignores the graph'
        problem = `Accuracy with edges ${fmt(num(e.acc_full))}, without edges ${fmt(num(e.acc_ablated))}.`
      }
      cards.push({
        kind: 'repair', source, fromHash: from, toHash: to || null, fromVersion: ver.get(from) ?? 1, toVersion: to ? ver.get(to) ?? null : null,
        problemTitle: title, problem, search, outcome: outcomeOf(events, claimId, to || null),
      })
    }
  }
  return cards
}

// ---- line diff (LCS), only changed lines with 2 lines of context ----
export type DiffLine = { t: 'ctx' | 'add' | 'del'; text: string }
export type Hunk = DiffLine[]

export function lineDiff(a: string, b: string, context = 2): Hunk[] {
  const x = a.replace(/\n$/, '').split('\n')
  const y = b.replace(/\n$/, '').split('\n')
  const n = x.length, m = y.length
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1))
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1])
  const all: DiffLine[] = []
  let i = 0, j = 0
  while (i < n && j < m) {
    if (x[i] === y[j]) { all.push({ t: 'ctx', text: x[i] }); i++; j++ }
    else if (L[i + 1][j] >= L[i][j + 1]) all.push({ t: 'del', text: x[i++] })
    else all.push({ t: 'add', text: y[j++] })
  }
  while (i < n) all.push({ t: 'del', text: x[i++] })
  while (j < m) all.push({ t: 'add', text: y[j++] })
  const keep = new Array(all.length).fill(false)
  all.forEach((l, k) => { if (l.t !== 'ctx') for (let d = -context; d <= context; d++) if (all[k + d]) keep[k + d] = true })
  const hunks: Hunk[] = []
  let cur: Hunk | null = null
  all.forEach((l, k) => {
    if (keep[k]) { if (!cur) { cur = []; hunks.push(cur) } cur.push(l) } else cur = null
  })
  return hunks
}

// "Play this check in 30 seconds" as a film: six scenes built only from the data of one real run. Pure functions: the scene plan, what each
// scene shows, and the sentences. Nothing here is invented; a scene whose data is missing is left out.
import type { Event, Json } from './data'
import { splitFormulas } from './latex.ts'
import { buildStory, lineDiff, type Card, type DiffLine } from './repairStory.ts'

export const FILM_MS = 30_000

export type SceneId = 'tavily' | 'code' | 'repair' | 'sandbox' | 'harness' | 'community' | 'verdict'
export type Scene = { id: SceneId; start: number; end: number }

// Relative length of each scene; the plan is scaled to FILM_MS, so a run without a repair still lasts 30 s.
const WEIGHT: Record<SceneId, number> = { tavily: 6, code: 5, repair: 5, sandbox: 5, harness: 6, community: 4, verdict: 3 }
const ORDER: SceneId[] = ['tavily', 'code', 'repair', 'sandbox', 'harness', 'community', 'verdict']

export function scenePlan(present: Set<SceneId>, lengthMs = FILM_MS): Scene[] {
  const ids = ORDER.filter((id) => present.has(id))
  const total = ids.reduce((n, id) => n + WEIGHT[id], 0)
  let at = 0
  return ids.map((id, i) => {
    const end = i === ids.length - 1 ? lengthMs : at + Math.round((WEIGHT[id] / total) * lengthMs)
    const s = { id, start: at, end }
    at = end
    return s
  })
}

// Scene index and progress inside it (0..1) at time `ms`.
export function sceneAt(plan: Scene[], ms: number): { index: number; p: number } {
  if (plan.length === 0) return { index: 0, p: 0 }
  const t = Math.max(0, Math.min(ms, plan[plan.length - 1].end))
  let index = plan.findIndex((s) => t < s.end)
  if (index < 0) index = plan.length - 1
  const s = plan[index]
  return { index, p: Math.min(1, Math.max(0, (t - s.start) / (s.end - s.start))) }
}

// ---- data of the scenes ----
export const domainOf = (url: string) => { try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url } }

export function addressBar(url: string): string {
  try { const u = new URL(url); return `${u.hostname.replace(/^www\./, '')}${u.pathname === '/' ? '' : u.pathname}` } catch { return url }
}

// Domains Tavily was asked about in this run (tavily_call targets) and the page the claim was read from; each once, in order.
// Only the calls that read the paper and its code page (roles 2 and 4); the later search of public discussion (roles 3 and 6) is its own scene.
export const isPaperCall = (e: Event) => e.type === 'tavily_call' && e.role !== 3 && e.role !== 6

export function sourceDomains(events: Event[], evidenceUrl?: string): string[] {
  const urls = [evidenceUrl, ...events.filter((e) => isPaperCall(e) && typeof e.target === 'string' && /^https?:/.test(e.target)).map((e) => e.target as string)]
  const out: string[] = []
  for (const u of urls) if (u) { const d = domainOf(u); if (!out.includes(d)) out.push(d) }
  return out
}

// First lines of code worth showing: no blank lines, no comment-only lines.
export function codeLines(code: string, n = 14, width = 78): string[] {
  return code.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#')).slice(0, n).map((l) => (l.length > width ? l.slice(0, width - 1) + '…' : l))
}

// Markdown markers out of a plan line: bold, code ticks, heading hashes, list markers. Nothing else is changed.
export const stripMarkdown = (l: string) => l.replace(/\*\*|__|`/g, '').replace(/^\s*(?:[-*+]|\d+[.)])\s+/, '').replace(/^#+\s*/, '').trim()

const isTitle = (l: string) => /^\s*#+\s/.test(l) || /^\s*(\*\*|__)[^*_]+(\*\*|__):?\s*$/.test(l)
const isItem = (l: string) => /^\s*(?:[-*+]|\d+[.)])\s+\S/.test(l)

// Cut at the last space before the limit, never inside a word or a formula.
function cut(l: string, width: number) {
  if (l.length <= width) return l
  const at = l.lastIndexOf(' ', width - 1)
  return l.slice(0, at > width / 2 ? at : width - 1).trimEnd() + '…'
}

// The plan for the film: a title (small label) and the first steps. A heading or an all-bold first line is the title, not a step; steps are the
// list items (or, in a plan without a list, the first longer lines).
export function planItems(plan: string, n = 3, width = 110): { title: string | null; steps: string[]; numbers: number[] } {
  const lines = plan.split('\n').filter((l) => l.trim())
  const t = lines.find(isTitle)
  const rest = lines.filter((l) => l !== t)
  const items = rest.filter(isItem)
  const chosen = (items.length ? items : rest.filter((l) => stripMarkdown(l).length > 12)).filter((l) => stripMarkdown(l)).slice(0, n)
  // the number of a step is the one written in the plan ("3. ..."), else its position; the agent's code comments refer to these numbers
  const numbers = chosen.map((l, i) => Number(l.match(/^\s*(\d+)[.)]\s/)?.[1] ?? i + 1))
  return { title: t ? cut(stripMarkdown(t), 70) : null, steps: chosen.map((l) => cut(stripMarkdown(l), width)), numbers }
}

// Text for display: "^(-1/2)" and "^2" become superscripts, nothing else is interpreted. A word that holds a formula stays in one piece (no line break
// inside it); spaces between words, such as around arrows, can break.
export type MathPart = { kind: 'text' | 'sup'; text: string }
export type MathWord = { formula: boolean; parts: MathPart[] }
export function mathWords(text: string): MathWord[] {
  return text.split(/(\s+)/).filter((w) => w !== '').map((w) => {
    if (/^\s+$/.test(w)) return { formula: false, parts: [{ kind: 'text' as const, text: w }] }
    const parts: MathPart[] = []
    let last = 0
    for (const m of w.matchAll(/\^\(([^)]*)\)|\^(-?[A-Za-z0-9.]+)/g)) {
      if (m.index! > last) parts.push({ kind: 'text', text: w.slice(last, m.index) })
      parts.push({ kind: 'sup', text: m[1] ?? m[2] })
      last = m.index! + m[0].length
    }
    if (last < w.length) parts.push({ kind: 'text', text: w.slice(last) })
    return { formula: parts.some((p) => p.kind === 'sup'), parts }
  })
}

// How much of a text is typed at progress p (characters), for typing animations; p is clamped.
export const typed = (text: string, p: number) => text.slice(0, Math.round(text.length * Math.min(1, Math.max(0, p))))

// The claim as the paper states it: the quote, the table and the page it comes from. Null without evidence.
export function claimEvidence(claim: Json): { quote: string; table: string | null; url: string | null; version: string | null } | null {
  const ev = claim?.target_evidence
  if (!ev?.quote) return null
  return { quote: String(ev.quote), table: ev.table ?? null, url: ev.url ?? null, version: ev.arxiv_version ?? null }
}

// Sandbox branches: one per seed, with the status of that seed's last attempt that counts (a valid measurement wins).
export type Branch = { seed: number; ok: boolean }
export function branches(events: Event[]): Branch[] {
  const by = new Map<number, Branch>()
  for (const e of events) {
    if (e.type !== 'attempt' || typeof e.seed !== 'number') continue
    const ok = typeof e.measured === 'number'
    const prev = by.get(e.seed)
    if (!prev || ok || !prev.ok) by.set(e.seed, { seed: e.seed, ok: ok || (prev?.ok ?? false) })
  }
  return [...by.values()].sort((a, b) => a.seed - b.seed)
}

// Positions (0..1 on the x axis) of the branch nodes under the snapshot.
export const branchX = (i: number, n: number) => (n <= 1 ? 0.5 : 0.06 + (0.88 * i) / (n - 1))

// Valid measurements in the order the harness received them.
export const measurements = (events: Event[], claimId: string): number[] =>
  events.filter((e) => e.type === 'attempt' && e.claim_id === claimId && typeof e.measured === 'number').map((e) => e.measured as number)

// The dossier at a moment of the harness scene: the first k measurements and, once the scene is nearly done, the verdict (mean line).
export function dossierAt(dossier: Json, claimId: string, events: Event[], k: number, withVerdict: boolean): Json {
  const vals = events.filter((e) => e.type === 'attempt' && e.claim_id === claimId && typeof e.measured === 'number').slice(0, k)
  return { ...dossier, attempts: vals.map((e) => ({ claim_id: claimId, seed: e.seed, measured: e.measured })), verdicts: withVerdict ? dossier.verdicts : [] }
}

// ---- the repair scene ----
export type RepairScene = {
  card: Extract<Card, { kind: 'repair' }>
  sentence: string
  version: { from: number; to: number | null }
}

export function repairScene(events: Event[], claimId: string): RepairScene | null {
  const card = buildStory(events, claimId).find((c): c is Extract<Card, { kind: 'repair' }> => c.kind === 'repair')
  if (!card) return null
  const sentence = card.source === 'error'
    ? card.search ? 'The code failed. The agent reads the error, searches the web and fixes it.' : 'The code failed. The agent reads the error and fixes it.'
    : card.source === 'fault' ? 'The first result was no better than a trivial guess, so the agent fixes its own code.'
      : 'The first code ignored the graph, so the agent fixes its own code.'
  return { card, sentence, version: { from: card.fromVersion, to: card.toVersion } }
}

// The changed lines of a diff, for the flash: removed and added lines, at most `n`.
export function diffFlash(a: string, b: string, n = 4): DiffLine[] {
  return lineDiff(a, b, 0).flat().filter((l) => l.t !== 'ctx' && l.text.trim()).slice(0, n)
}

// ---- which scenes this run has ----
export function presentScenes(o: { events: Event[]; claim: Json; hasPlan: boolean; hasCode: boolean; hasVerdict: boolean; repair: boolean; community?: boolean }): Set<SceneId> {
  const s = new Set<SceneId>()
  if (claimEvidence(o.claim) || o.events.some(isPaperCall)) s.add('tavily')
  if (o.hasPlan || o.hasCode) s.add('code')
  if (o.repair) s.add('repair')
  const br = branches(o.events)
  if (br.length > 0) s.add('sandbox')
  if (measurements(o.events, String(o.claim?.id)).length > 0) s.add('harness')
  if (o.community) s.add('community')
  if (o.hasVerdict) s.add('verdict')
  return s
}

// Sentences, one per scene. Each is true for the run it is shown for (the scenes it depends on are only present when their data is).
export const SENTENCE: Record<Exclude<SceneId, 'repair' | 'verdict' | 'tavily' | 'community'>, (n?: number) => string> = {
  code: () => 'Nemotron plans the experiment and writes the code.',
  sandbox: (n) => `Each random seed runs in its own sandbox branch${n ? `: ${n} branches` : ''}.`,
  harness: () => 'Our harness scores every run. The agent never sees the true answers.',
}

export const ACTOR: Record<SceneId, string> = {
  tavily: 'Tavily', code: 'Nemotron on Token Factory', repair: 'Nemotron + Tavily', sandbox: 'Nebius Sandbox', harness: 'Our harness', community: 'Tavily + Nemotron', verdict: 'Our harness',
}

export const SCENE_LABEL: Record<SceneId, string> = {
  tavily: 'Paper', code: 'Code', repair: 'Repair', sandbox: 'Sandbox', harness: 'Check', community: 'Others', verdict: 'Verdict',
}

// "The real run took 41 minutes" (first to last event).
export function durationText(events: Event[]): string | null {
  const ts = events.map((e) => e.ts).filter((t) => typeof t === 'number' && Number.isFinite(t))
  if (ts.length < 2) return null
  const s = Math.max(...ts) - Math.min(...ts)
  if (s < 90) return `${Math.max(1, Math.round(s))} seconds`
  const min = Math.round(s / 60)
  if (min < 90) return `${min} minutes`
  return `${Math.floor(min / 60)} h ${min % 60} min`
}

export const eventsForClaim = (events: Event[], claimId: string): Event[] => events.filter((e) => e.claim_id === undefined || e.claim_id === claimId)

// The first scene says "finds the number" only if the run really has the quote it was found in.
export const tavilySentence = (hasEvidence: boolean) => (hasEvidence ? 'Tavily reads the paper and finds the number it reports.' : 'Tavily reads the paper and the page of its code.')

// ---- plan cards ----
// "Model: 2-layer GCN: ..." -> category "Model" and the rest. A category is the text before the first colon if it is short and plain; otherwise none.
export function splitCategory(step: string): { category: string | null; body: string } {
  const m = step.match(/^([^:]{2,40}?):\s+(\S[\s\S]*)$/)
  if (!m || /[=^()$→]|->/.test(m[1])) return { category: null, body: step }
  return { category: m[1].trim(), body: m[2].trim() }
}

const ARROW = /→|⟶|->/
// An arrow chain "X → GCN(16, ReLU) → softmax" as a pipeline: the text in front of it (up to the last colon) and the nodes. Arrows inside
// brackets do not split. Null if there is no chain of at least two nodes, or a node is long (then it is prose, not a pipeline).
export function pipelineOf(body: string): { lead: string; nodes: string[] } | null {
  const parts: string[] = []
  let depth = 0, from = 0
  for (let i = 0; i < body.length; i++) {
    const c = body[i]
    if ('([{'.includes(c)) depth++
    else if (')]}'.includes(c)) depth = Math.max(0, depth - 1)
    else if (depth === 0) {
      const m = body.slice(i).match(/^(→|⟶|->)/)
      if (m) { parts.push(body.slice(from, i)); i += m[0].length - 1; from = i + 1 }
    }
  }
  if (parts.length === 0) return null
  parts.push(body.slice(from))
  const first = parts[0].trim()
  const colon = first.lastIndexOf(': ')
  const lead = colon >= 0 ? first.slice(0, colon + 1) : ''
  const nodes = [colon >= 0 ? first.slice(colon + 2).trim() : first, ...parts.slice(1).map((p) => p.trim())].filter(Boolean)
  if (nodes.length < 2 || nodes.some((n) => n.length > 48) || lead.length > 70 || !ARROW.test(body)) return null
  return { lead, nodes }
}

// ---- plan step <-> code ----
// The agent's own mapping comment at the top of its code ("# 1. Model -> NLinear class"). Only a mapping whose category matches a shown plan
// step AND whose named function, class or constant really is defined in the code counts; nothing else is linked.
export type CodeLink = { step: number; name: string; lines: string[] }
const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '')

export type MapEntry = { category: string | null; number: number | null; target: string }
// Two forms of the agent's comment: "# 1. Model -> NLinear class" (by category) and "# Plan step 3 -> implemented in: normalize_x" (by number).
export function mappingComments(code: string): MapEntry[] {
  const out: MapEntry[] = []
  for (const l of code.split('\n').slice(0, 80)) {
    const byNumber = l.match(/^\s*#\s*Plan step\s+(\d+)\s*->\s*implemented in:?\s*(.+?)\s*$/i)
    if (byNumber) { out.push({ category: null, number: Number(byNumber[1]), target: byNumber[2] }); continue }
    const byCategory = l.match(/^\s*#\s*\d+\.?\s*(.+?)\s*->\s*(.+?)\s*$/)
    if (byCategory) out.push({ category: byCategory[1], number: null, target: byCategory[2] })
  }
  return out
}

export function definitionLine(code: string, name: string): number {
  const re = new RegExp(`^\\s*(?:(?:async\\s+)?def|class)\\s+${name}\\b|^${name}\\s*(?::[^=\\n]+)?=`)
  return code.split('\n').findIndex((l) => re.test(l))
}

export function planCodeLinks(code: string, steps: { category: string | null; number: number }[], linesPer = 3, width = 78): CodeLink[] {
  const map = mappingComments(code)
  const lines = code.split('\n')
  const out: CodeLink[] = []
  steps.forEach((st, step) => {
    const m = map.find((x) => (x.number != null ? x.number === st.number : x.category != null && st.category != null && norm(x.category) === norm(st.category)))
    if (!m) return
    for (const name of [...new Set(m.target.match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? [])]) {
      const at = definitionLine(code, name)
      if (at < 0) continue
      const snippet = lines.slice(at, at + linesPer + 3).filter((l) => l.trim() && !l.trim().startsWith('#')).slice(0, linesPer).map((l) => (l.length > width ? l.slice(0, width - 1) + '…' : l))
      if (snippet.length) out.push({ step, name, lines: snippet })
      return
    }
  })
  return out
}

// How much room a plan card needs (1 = one text line plus its label): formulas and long arrow chains take more. Used to show as many
// steps as fit the column without cutting a card; the steps shown are always the first ones, in order.
export function cardWeight(step: string): number {
  const { body } = splitCategory(step)
  const maths = splitFormulas(body).filter((s) => s.kind === 'math').length
  if (maths > 0) return 1 + 0.45 * maths
  const pipe = pipelineOf(body)
  if (pipe) return 1 + (pipe.nodes.length >= 4 ? 0.7 : 0.35)
  return body.length > 70 ? 1.35 : 1
}

export function fitSteps(steps: string[], budget: number): string[] {
  const out: string[] = []
  let used = 0
  for (const s of steps) {
    const w = cardWeight(s)
    if (out.length > 0 && used + w > budget) break
    out.push(s)
    used += w
  }
  return out
}

// The community scene: the public notes the check found (reported only, never part of the verdict). Null if the run has no community check.
export type Note = { statement: string; quote: string; host: string; kind: string }
export function communityScene(dossier: Json): { notes: Note[]; total: number; sentence: string } | null {
  const c = dossier?.community
  if (!c) return null
  const all: Json[] = c.findings ?? []
  const n = all.length
  const sentence = n > 0
    ? `Tavily also searches public discussion of the paper: ${n} note${n === 1 ? '' : 's'} found, each with a quote from the page. They never change the verdict.`
    : 'Tavily also searched public discussion of the paper and found nothing. It never changes the verdict.'
  const host = (u: string) => { try { return new URL(u).hostname.replace(/^www\./, '') } catch { return u } }
  const notes = all.slice(0, 2).map((f) => ({ statement: String(f.statement), quote: String(f.quote).length > 150 ? String(f.quote).slice(0, 149) + '…' : String(f.quote), host: host(String(f.url)), kind: String(f.kind ?? 'other') }))
  return { notes, total: n, sentence }
}

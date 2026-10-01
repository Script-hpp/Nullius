// "Play this check in 30 seconds" as a film: six scenes built only from the data of one real run. Pure functions: the scene plan, what each
// scene shows, and the sentences. Nothing here is invented; a scene whose data is missing is left out.
import type { Event, Json } from './data'
import { buildStory, lineDiff, type Card, type DiffLine } from './repairStory.ts'

export const FILM_MS = 30_000

export type SceneId = 'tavily' | 'code' | 'repair' | 'sandbox' | 'harness' | 'verdict'
export type Scene = { id: SceneId; start: number; end: number }

// Relative length of each scene; the plan is scaled to FILM_MS, so a run without a repair still lasts 30 s.
const WEIGHT: Record<SceneId, number> = { tavily: 6, code: 5, repair: 5, sandbox: 5, harness: 6, verdict: 3 }
const ORDER: SceneId[] = ['tavily', 'code', 'repair', 'sandbox', 'harness', 'verdict']

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
export function sourceDomains(events: Event[], evidenceUrl?: string): string[] {
  const urls = [evidenceUrl, ...events.filter((e) => e.type === 'tavily_call' && typeof e.target === 'string' && /^https?:/.test(e.target)).map((e) => e.target as string)]
  const out: string[] = []
  for (const u of urls) if (u) { const d = domainOf(u); if (!out.includes(d)) out.push(d) }
  return out
}

// First lines of code worth showing: no blank lines, no comment-only lines.
export function codeLines(code: string, n = 14, width = 78): string[] {
  return code.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#')).slice(0, n).map((l) => (l.length > width ? l.slice(0, width - 1) + '…' : l))
}

// First bullets of the plan, markdown stripped.
export function planBullets(plan: string, n = 3, width = 84): string[] {
  const clean = (l: string) => l.replace(/\*\*|__|`/g, '').replace(/^\s*(?:[-*]|\d+[.)])\s*/, '').replace(/^#+\s*/, '').trim()
  return plan.split('\n').map(clean).filter((l) => l.length > 12).slice(0, n).map((l) => (l.length > width ? l.slice(0, width - 1) + '…' : l))
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
export function presentScenes(o: { events: Event[]; claim: Json; hasPlan: boolean; hasCode: boolean; hasVerdict: boolean; repair: boolean }): Set<SceneId> {
  const s = new Set<SceneId>()
  if (claimEvidence(o.claim) || o.events.some((e) => e.type === 'tavily_call')) s.add('tavily')
  if (o.hasPlan || o.hasCode) s.add('code')
  if (o.repair) s.add('repair')
  const br = branches(o.events)
  if (br.length > 0) s.add('sandbox')
  if (measurements(o.events, String(o.claim?.id)).length > 0) s.add('harness')
  if (o.hasVerdict) s.add('verdict')
  return s
}

// Sentences, one per scene. Each is true for the run it is shown for (the scenes it depends on are only present when their data is).
export const SENTENCE: Record<Exclude<SceneId, 'repair' | 'verdict' | 'tavily'>, (n?: number) => string> = {
  code: () => 'Nemotron plans the experiment and writes the code.',
  sandbox: (n) => `Each random seed runs in its own sandbox branch${n ? `: ${n} branches` : ''}.`,
  harness: () => 'Our harness scores every run. The agent never sees the true answers.',
}

export const ACTOR: Record<SceneId, string> = {
  tavily: 'Tavily', code: 'Nemotron on Token Factory', repair: 'Nemotron + Tavily', sandbox: 'Nebius Sandbox', harness: 'Our harness', verdict: 'Our harness',
}

export const SCENE_LABEL: Record<SceneId, string> = {
  tavily: 'Paper', code: 'Code', repair: 'Repair', sandbox: 'Sandbox', harness: 'Check', verdict: 'Verdict',
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

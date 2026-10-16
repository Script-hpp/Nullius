// What the UI says when live checks are paused. Pure functions (no React), tested in livestatus.test.ts.
// The server answers with a reason, never with balances; the UI never shows an account figure.
export type PauseReason = 'budget_floor' | 'tavily_floor' | 'daily_limit' | 'busy'
export type Kind = 'reproduction' | 'discovery'
export type KindStatus = { live: boolean; reason: PauseReason | null; reset_at: string | null }
export type LiveStatusInfo = KindStatus & { kinds?: Partial<Record<Kind, KindStatus>> }

const isReason = (r: unknown): r is PauseReason => r === 'budget_floor' || r === 'tavily_floor' || r === 'daily_limit' || r === 'busy'

// Why live runs are closed, in everyday words. No balances, no account figures.
const WHY: Record<Exclude<PauseReason, 'busy'>, string> = {
  budget_floor: 'the demo budget for live runs is used up',
  tavily_floor: 'the search allowance for this demo is used up',
  daily_limit: 'the daily limit is reached',
}

// Short label for a disabled control or a card line: "Live check closed today".
export function closedShort(reason: PauseReason | null | undefined, target: 'check' | 'search' = 'check'): string {
  const what = target === 'check' ? 'Live check' : 'Live search'
  if (reason === 'busy') return `${what}: another run is active`
  return reason === 'daily_limit' ? `${what} closed today` : `${what} closed`
}

// "Live checks are closed: the daily limit is reached. Open the recorded result instead."
export function pausedText(reason: PauseReason | null | undefined, target: 'check' | 'search' = 'check'): string {
  if (reason === 'busy') return 'Another check is running. Try again in a few minutes, or open the recorded result.'
  const why = reason ? WHY[reason] : 'a limit is reached'
  return `Live ${target === 'check' ? 'checks are' : 'search is'} closed: ${why}. Open the recorded ${target === 'check' ? 'result' : 'search'} instead.`
}

// "Live checks resume around 00:00 UTC." only for the daily limit.
export function resumeText(reason: PauseReason | null | undefined, resetAt: string | null | undefined): string | null {
  if (reason !== 'daily_limit' || !resetAt) return null
  const t = new Date(resetAt)
  if (Number.isNaN(t.getTime())) return null
  return `Live checks reopen around ${t.toISOString().slice(11, 16)} UTC.`
}

export function statusFor(s: LiveStatusInfo | null, kind: Kind): KindStatus | null {
  if (!s) return null
  return s.kinds?.[kind] ?? { live: s.live, reason: s.reason, reset_at: s.reset_at }
}

// A paused status that is not just "another run is active" (that one has its own hint).
export const isPaused = (k: KindStatus | null): boolean => !!k && !k.live && k.reason !== 'busy' && k.reason !== null

export class StartRefused extends Error {
  reason: PauseReason | null
  constructor(message: string, reason: PauseReason | null) { super(message); this.reason = reason }
}

// Text for a failed start (HTTP status and the JSON body, if any). 503/429 with a known reason are calm notices, not errors.
export function startErrorText(status: number, body: unknown, target: 'check' | 'search' = 'check'): { text: string; reason: PauseReason | null } {
  const b = (body && typeof body === 'object' ? body : {}) as { reason?: unknown; detail?: unknown }
  if (b.reason === 'rate_limited') return { text: 'Too many requests from this connection. Try again in a minute.', reason: null }
  const reason = isReason(b.reason) ? b.reason : null
  if (reason) return { text: pausedText(reason, target), reason }
  if (status === 503 || status === 429) return { text: pausedText(status === 503 ? 'budget_floor' : 'daily_limit', target), reason: null }
  const detail = typeof b.detail === 'string' ? b.detail : ''
  if (status === 400 || status === 409) return { text: detail || 'This entry cannot be run.', reason: null }   // 409: a duplicate find of an offered claim
  if (status === 404) return { text: 'The live backend does not know this entry.', reason: null }
  return { text: `The live backend answered with an error (${status}).`, reason: null }
}

export type StartBlock = { blocked: false } | { blocked: true; short: string; long: string }

// Can a live start button work right now? When not, the reason in plain words (short for a card line, long for a banner).
// `available` = the live API answers; `kind` = status of this kind of run; `busy` = another job is running; `runnable` = eligibility of this entry.
export function startBlock(i: { available: boolean; kind: KindStatus | null; busy: boolean; runnable?: boolean; target?: 'check' | 'search'; resetAt?: string | null }): StartBlock {
  const target = i.target ?? 'check'
  if (!i.available) return target === 'check'
    ? { blocked: true, short: 'Live checks are off on this site', long: 'Live checks are off on this site. Recorded results open as usual.' }
    : { blocked: true, short: 'Live search is off on this site', long: 'Live search is off on this site. The recorded search is open.' }
  if (i.kind && !i.kind.live && i.kind.reason && i.kind.reason !== 'busy') {
    const resume = resumeText(i.kind.reason, i.kind.reset_at)
    return { blocked: true, short: closedShort(i.kind.reason, target), long: `${pausedText(i.kind.reason, target)}${resume ? ` ${resume}` : ''}` }
  }
  if (i.busy || i.kind?.reason === 'busy') return { blocked: true, short: closedShort('busy', target), long: pausedText('busy', target) }
  if (i.runnable === false) return { blocked: true, short: 'No live check for this entry', long: 'This entry cannot be rebuilt here. Open its recorded result instead.' }
  return { blocked: false }
}

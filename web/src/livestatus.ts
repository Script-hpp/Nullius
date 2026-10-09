// What the UI says when live checks are paused. Pure functions (no React), tested in livestatus.test.ts.
// The server answers with a reason, never with balances; the UI never shows an account figure.
export type PauseReason = 'budget_floor' | 'tavily_floor' | 'daily_limit' | 'busy'
export type Kind = 'reproduction' | 'discovery'
export type KindStatus = { live: boolean; reason: PauseReason | null; reset_at: string | null }
export type LiveStatusInfo = KindStatus & { kinds?: Partial<Record<Kind, KindStatus>> }

const CAUSE: Record<Exclude<PauseReason, 'busy'>, string> = {
  budget_floor: 'budget limit',
  tavily_floor: 'budget limit',
  daily_limit: 'daily limit',
}

const isReason = (r: unknown): r is PauseReason => r === 'budget_floor' || r === 'tavily_floor' || r === 'daily_limit' || r === 'busy'

// "Live checks are paused (daily limit). Here is the recorded run of this check."
export function pausedText(reason: PauseReason | null | undefined, target: 'check' | 'search' = 'check'): string {
  if (reason === 'busy') return 'Another check is running. Try again in a few minutes.'
  const cause = reason ? CAUSE[reason] : 'limit'
  return `Live checks are paused (${cause}). Here is the recorded run of this ${target}.`
}

// "Live checks resume around 00:00 UTC." only for the daily limit.
export function resumeText(reason: PauseReason | null | undefined, resetAt: string | null | undefined): string | null {
  if (reason !== 'daily_limit' || !resetAt) return null
  const t = new Date(resetAt)
  if (Number.isNaN(t.getTime())) return null
  return `Live checks resume around ${t.toISOString().slice(11, 16)} UTC.`
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
  if (status === 400) return { text: detail || 'This entry cannot be run.', reason: null }
  if (status === 404) return { text: 'The live backend does not know this entry.', reason: null }
  return { text: `The live backend answered with an error (${status}).`, reason: null }
}

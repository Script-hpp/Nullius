import test from 'node:test'
import assert from 'node:assert/strict'
import { closedShort, isPaused, pausedText, startBlock, resumeText, startErrorText, statusFor } from './livestatus.ts'

test('text per reason: why it is closed and what to open instead', () => {
  assert.equal(pausedText('daily_limit'), 'Live checks are closed: the daily limit is reached. Open the recorded result instead.')
  assert.equal(pausedText('budget_floor'), 'Live checks are closed: the demo budget for live runs is used up. Open the recorded result instead.')
  assert.equal(pausedText('tavily_floor'), 'Live checks are closed: the search allowance for this demo is used up. Open the recorded result instead.')
  assert.equal(pausedText('daily_limit', 'search'), 'Live search is closed: the daily limit is reached. Open the recorded search instead.')
  assert.match(pausedText('busy'), /Another check is running/)
})

test('no balance or account figure in any text', () => {
  for (const r of ['budget_floor', 'tavily_floor', 'daily_limit', 'busy'] as const) {
    assert.doesNotMatch(pausedText(r), /\$|credit|usd/i)
    assert.doesNotMatch(closedShort(r), /\$|credit|usd/i)
  }
})

test('every blocked start says why and points to the next action', () => {
  const off = { available: true, busy: false }
  for (const reason of ['budget_floor', 'tavily_floor', 'daily_limit'] as const) {
    const b = startBlock({ ...off, kind: { live: false, reason, reset_at: null } })
    assert.ok(b.blocked && b.short.length > 0 && /recorded/i.test(b.long), reason)
  }
  const busy = startBlock({ ...off, busy: true, kind: null })
  assert.ok(busy.blocked && /another/i.test(busy.short) && /recorded/i.test(busy.long))
  const closed = startBlock({ available: false, kind: null, busy: false })
  assert.ok(closed.blocked && /recorded/i.test(closed.long))
  const unrunnable = startBlock({ ...off, kind: { live: true, reason: null, reset_at: null }, runnable: false })
  assert.ok(unrunnable.blocked && /recorded/i.test(unrunnable.long))
  assert.deepEqual(startBlock({ ...off, kind: { live: true, reason: null, reset_at: null }, runnable: true }), { blocked: false })
})

test('resume time only for the daily limit', () => {
  assert.equal(resumeText('daily_limit', '2026-10-06T22:00:00Z'), 'Live checks reopen around 22:00 UTC.')
  assert.equal(resumeText('budget_floor', '2026-10-06T22:00:00Z'), null)
  assert.equal(resumeText('daily_limit', null), null)
  assert.equal(resumeText('daily_limit', 'garbage'), null)
})

test('start errors: 503 and 429 are notices', () => {
  assert.deepEqual(startErrorText(503, { reason: 'budget_floor' }), { text: pausedText('budget_floor'), reason: 'budget_floor' })
  assert.equal(startErrorText(429, { reason: 'busy' }).reason, 'busy')
  assert.equal(startErrorText(409, { detail: 'Same claim as an earlier find.' }).text, 'Same claim as an earlier find.')   // duplicate find refused with its reason
  assert.equal(startErrorText(429, { reason: 'daily_limit' }, 'search').text, pausedText('daily_limit', 'search'))
  assert.equal(startErrorText(503, {}).text, pausedText('budget_floor'))
  assert.equal(startErrorText(400, { detail: 'Not runnable: x' }).text, 'Not runnable: x')
  assert.match(startErrorText(500, null).text, /error \(500\)/)
})

test('status per kind and paused detection', () => {
  const s = { live: true, reason: null, reset_at: null, kinds: { reproduction: { live: false, reason: 'daily_limit' as const, reset_at: null }, discovery: { live: true, reason: null, reset_at: null } } }
  assert.equal(isPaused(statusFor(s, 'reproduction')), true)
  assert.equal(isPaused(statusFor(s, 'discovery')), false)
  assert.equal(isPaused(statusFor({ live: false, reason: 'busy', reset_at: null }, 'discovery')), false)
  assert.equal(isPaused(statusFor(null, 'discovery')), false)
})

test('rate limited start is a calm notice, not a pause reason', () => {
  const r = startErrorText(429, { reason: 'rate_limited' })
  assert.equal(r.reason, null)
  assert.match(r.text, /Too many requests/)
})

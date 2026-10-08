import test from 'node:test'
import assert from 'node:assert/strict'
import { isPaused, pausedText, resumeText, startErrorText, statusFor } from './livestatus.ts'

test('text per reason', () => {
  assert.equal(pausedText('daily_limit'), 'Live checks are paused (daily limit). Here is the recorded run of this check.')
  assert.equal(pausedText('budget_floor'), 'Live checks are paused (budget limit). Here is the recorded run of this check.')
  assert.equal(pausedText('tavily_floor'), 'Live checks are paused (budget limit). Here is the recorded run of this check.')
  assert.equal(pausedText('daily_limit', 'search'), 'Live checks are paused (daily limit). Here is the recorded run of this search.')
  assert.match(pausedText('busy'), /Another check is running/)
})

test('no balance or account figure in any text', () => {
  for (const r of ['budget_floor', 'tavily_floor', 'daily_limit', 'busy'] as const) assert.doesNotMatch(pausedText(r), /\$|credit|usd/i)
})

test('resume time only for the daily limit', () => {
  assert.equal(resumeText('daily_limit', '2026-10-06T22:00:00Z'), 'Live checks resume around 22:00 UTC.')
  assert.equal(resumeText('budget_floor', '2026-10-06T22:00:00Z'), null)
  assert.equal(resumeText('daily_limit', null), null)
  assert.equal(resumeText('daily_limit', 'garbage'), null)
})

test('start errors: 503 and 429 are notices', () => {
  assert.deepEqual(startErrorText(503, { reason: 'budget_floor' }), { text: pausedText('budget_floor'), reason: 'budget_floor' })
  assert.equal(startErrorText(429, { reason: 'busy' }).reason, 'busy')
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

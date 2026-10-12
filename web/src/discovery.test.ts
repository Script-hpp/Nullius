import test from 'node:test'
import assert from 'node:assert/strict'
import { deriveView, tickedRuntime, parseLiveEvent, type DEvent } from './discovery.ts'

test('early server errors without timestamps remain visible and cannot create a NaN runtime', () => {
  const view = deriveView([{ type: 'error', message: 'Missing configuration' } as DEvent])
  assert.deepEqual(view.notes, [{ kind: 'error', message: 'Missing configuration' }])
  assert.equal(view.tEnd - view.t0, 0)
  assert.equal(tickedRuntime(NaN, 1000, 1, true), 1)
})

test('control events without timestamps do not poison recorded event timing', () => {
  const view = deriveView([
    { type: 'pending', ts: 100 },
    { type: 'tavily_call', ts: 105, duration_ms: 2000, endpoint: 'search' },
    { type: 'done' } as DEvent,
  ])
  assert.equal(view.t0, 100)
  assert.equal(view.tEnd, 105)
  assert.equal(view.tavilyCalls, 1)
})


test('live events preserve valid timestamps and normalize early error events', () => {
  assert.equal(parseLiveEvent('{"type":"error","message":"Missing configuration"}', 123)?.ts, 123)
  assert.equal(parseLiveEvent('{"type":"done","ts":120}', 123)?.ts, 120)
  assert.equal(parseLiveEvent('{"type":"error","ts":"bad"}', 123)?.ts, 123)
  assert.equal(parseLiveEvent('keepalive'), null)
  assert.equal(parseLiveEvent('null'), null)
  assert.equal(parseLiveEvent('{}'), null)
})

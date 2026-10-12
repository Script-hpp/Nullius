import test from 'node:test'
import assert from 'node:assert/strict'
import { tickedRuntime } from './discovery.ts'

test('timer ticks between events while running', () => {
  assert.equal(tickedRuntime(10, 0, 1, true), 10)
  assert.equal(tickedRuntime(10, 1500, 1, true), 11.5)
  assert.equal(tickedRuntime(10, 500, 4, true), 12)
})

test('frozen at the event runtime when not running; never negative', () => {
  assert.equal(tickedRuntime(10, 5000, 1, false), 10)
  assert.equal(tickedRuntime(-3, 0, 1, true), 0)
})

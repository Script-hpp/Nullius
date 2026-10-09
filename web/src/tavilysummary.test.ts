import test from 'node:test'
import assert from 'node:assert/strict'
import { tavilySummary } from './tavilysummary.ts'

test('counts Tavily calls by role, in role order', () => {
  const s = tavilySummary([{ type: 'tavily_call', role: 4 }, { type: 'tavily_call', role: 1 }, { type: 'tavily_call', role: 4 }, { type: 'plan' }] as never)
  assert.equal(s.total, 3)
  assert.deepEqual(s.parts, ['find material ×1', 'setup details ×2'])
})

test('no calls: empty summary', () => {
  assert.deepEqual(tavilySummary([]), { total: 0, parts: [] })
})

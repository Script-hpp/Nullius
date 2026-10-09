import test from 'node:test'
import assert from 'node:assert/strict'
import { tavilyChip, tavilySummary } from './tavilysummary.ts'

test('counts Tavily calls by role, in role order', () => {
  const s = tavilySummary([{ type: 'tavily_call', role: 4 }, { type: 'tavily_call', role: 1 }, { type: 'tavily_call', role: 4 }, { type: 'plan' }] as never)
  assert.equal(s.total, 3)
  assert.deepEqual(s.parts, ['find material ×1', 'setup details ×2'])
})

test('no calls: empty summary', () => {
  assert.deepEqual(tavilySummary([]), { total: 0, parts: [] })
})

test('chip: says live only with a live source, none without calls', () => {
  const ev = [{ type: 'tavily_call', role: 2 }, { type: 'tavily_call', role: 4 }] as never
  assert.equal(tavilyChip(ev, [{ mode: 'live' }, { mode: 'cache' }]), '2 live Tavily calls')
  assert.equal(tavilyChip(ev, [{ mode: 'cache' }]), '2 Tavily calls')
  assert.equal(tavilyChip([{ type: 'tavily_call', role: 1 }] as never, undefined), '1 Tavily call')
  assert.equal(tavilyChip([], [{ mode: 'live' }]), null)
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { allSettled } from './loaded.ts'
import { groupRuns } from './group.ts'

const run = (paper: string, ts: string) => ({ id: `${paper}--${ts}`, paper_id: paper, title: paper, timestamp: ts })

test('not settled until every run is settled', () => {
  const runs = [run('a', '1'), run('b', '2')]
  assert.equal(allSettled(runs, new Set([runs[0].id])), false)
  assert.equal(allSettled(runs, new Set(runs.map((r) => r.id))), true)
  assert.equal(allSettled([], new Set()), true)
})

test('card key (latest run id) stays stable while the grouping refines', () => {
  const runs = [run('z', '1'), run('z', '2')]
  const before = groupRuns(runs, () => undefined)
  const after = groupRuns(runs, () => 'same_claim')
  assert.equal(before[0].latest.id, after[0].latest.id)
})

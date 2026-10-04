import test from 'node:test'
import assert from 'node:assert/strict'
import { groupRuns } from './group.ts'

const run = (paper: string, ts: string) => ({ id: `${paper}--${ts}`, paper_id: paper, title: paper, timestamp: ts })

test('runs of one claim become one card with the newest run first', () => {
  const runs = [run('sgc', '20260929-124736'), run('sgc', '20260929-125849'), run('kipf', '20260929-130133')]
  const g = groupRuns(runs, (r) => (r.paper_id === 'sgc' ? 'sgc_acc' : 'gcn_acc'))
  assert.equal(g.length, 2)
  const sgc = g.find((x) => x.key === 'sgc|sgc_acc')!
  assert.equal(sgc.latest.timestamp, '20260929-125849')
  assert.deepEqual(sgc.earlier.map((r) => r.timestamp), ['20260929-124736'])
})

test('no run is lost', () => {
  const runs = [run('a', '1'), run('a', '2'), run('a', '3'), run('b', '4')]
  const g = groupRuns(runs, () => 'c')
  assert.equal(g.reduce((n, x) => n + 1 + x.earlier.length, 0), runs.length)
})

test('different claims of one paper stay separate cards', () => {
  const g = groupRuns([run('z', '1'), run('z', '2')], (r) => (r.timestamp === '1' ? 'dlinear' : 'nlinear'))
  assert.equal(g.length, 2)
})

test('cards are ordered by their newest run', () => {
  const g = groupRuns([run('a', '5'), run('b', '9')], () => 'x')
  assert.deepEqual(g.map((x) => x.latest.paper_id), ['b', 'a'])
})

test('without a loaded claim the paper is the key', () => {
  const g = groupRuns([run('a', '1'), run('a', '2')], () => undefined)
  assert.equal(g.length, 1)
  assert.equal(g[0].key, 'a')
})

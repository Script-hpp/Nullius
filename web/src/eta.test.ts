import test from 'node:test'
import assert from 'node:assert/strict'
import { etaFromRuns, minutes, runTimings } from './eta.ts'

const run = (first: number | null, end: number | null) => [
  { ts: 1000, type: 'run_start' },
  ...(first == null ? [] : [{ ts: 1000 + first, type: 'attempt', measured: 0.8 }]),
  ...(end == null ? [] : [{ ts: 1000 + end, type: 'run_end' }]),
]

test('timings are read from run_start, first measured attempt and run_end', () => {
  assert.deepEqual(runTimings(run(120, 300)), { seed0: 120, total: 300 })
})

test('an attempt without a measurement does not count as seed 0', () => {
  const ev = [{ ts: 0, type: 'run_start' }, { ts: 50, type: 'attempt', measured: null }, { ts: 90, type: 'attempt', measured: 0.7 }, { ts: 200, type: 'run_end' }]
  assert.equal(runTimings(ev).seed0, 90)
})

test('no run_start gives no timings', () => {
  assert.deepEqual(runTimings([{ ts: 5, type: 'attempt', measured: 1 }]), { seed0: null, total: null })
})

test('median, range and count over finished runs; unfinished runs are ignored', () => {
  const eta = etaFromRuns([run(100, 200), run(300, 600), run(200, 400), run(50, null)])
  assert.deepEqual(eta, { n: 3, seed0: 200, total: 400, totalMin: 200, totalMax: 600 })
})

test('no finished run, no estimate', () => {
  assert.equal(etaFromRuns([run(10, null)]), null)
  assert.equal(etaFromRuns([]), null)
})

test('minutes formatting', () => {
  assert.equal(minutes(45), '45 s')
  assert.equal(minutes(180), '3 min')
})

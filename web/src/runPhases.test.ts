import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { buildRunModel } from './components/run/phases.ts'
import type { Event } from './data.ts'

// The live SARAF run completed candidate 0 while candidates 1 and 2 crashed
// on their own seed-0 branches. Arrival order interleaves those branches.
const start: Event[] = [
  { ts: 100, type: 'run_start', seeds: 3 },
  { ts: 101, type: 'claim_start', metric: 'mse', target: 0.371, tolerance: 0.01 },
  { ts: 101.5, type: 'plan', text: 'Train SARAF' },
  { ts: 102, type: 'code_written', candidate: 0, code_hash: 'a' },
  { ts: 103, type: 'code_written', candidate: 1, code_hash: 'b' },
  { ts: 104, type: 'attempt', candidate: 1, seed: 0, n: 1, code_hash: 'b', measured: null, error: 'exit code 1' },
  { ts: 105, type: 'repair', seed: 0, from_hash: 'b', error: 'exit code 1' },
  { ts: 106, type: 'repair_result', seed: 0, from_hash: 'b', to_hash: 'b2' },
  { ts: 107, type: 'code_text', code_hash: 'b2', code: 'repaired code' },
  { ts: 108, type: 'attempt', candidate: 0, seed: 0, n: 1, code_hash: 'a', measured: 0.827, seconds: 88 },
  { ts: 109, type: 'attempt', candidate: 1, seed: 0, n: 2, code_hash: 'b2', measured: 0.9 },
]
const selection: Event = { ts: 110, type: 'candidates', chosen: 0, chosen_hash: 'a', k: 2 }
const build = (events: Event[]) => buildRunModel(events, { running: true })

test('parallel candidates keep their scores in history until selection, then chart only selected seed 0', () => {
  const pending = build(start)
  assert.equal(pending.candidateSelectionPending, true)
  assert.equal(pending.stats, null)
  assert.match(pending.waiting!, /candidates in parallel/)
  assert.ok(pending.phases.flatMap(p => p.rows).some(r => r.title.includes('scored 0.827')))
  const selected = build([...start, selection])
  assert.equal(selected.selectedCandidate, 0)
  assert.equal(selected.stats!.mean, 0.827)
  assert.equal(selected.stats!.n, 1)
})

test('attempts, crashes and repaired code identify their candidate and code; code appears before its result', () => {
  const rows = build(start).phases.find(p => p.id === 'sandbox')!.rows
  assert.ok(rows.some(r => /Candidate 1 · Seed 0, attempt 1 · Code version 1 finished/.test(r.title)))
  assert.ok(rows.some(r => /Candidate 2 · Code version 2 crashed/.test(r.title)))
  const rewritten = rows.find(r => r.expand?.kind === 'code' && r.expand.hash === 'b2')!
  assert.match(rewritten.title, /^Candidate 2/)
  assert.equal(rewritten.ts, 107)
  assert.ok(rows.indexOf(rewritten) < rows.findIndex(r => r.raw[0]?.code_hash === 'b2'))
})

test('accepted repair replaces the selected seed measurement; rejected repository repair keeps it', () => {
  const repaired: Event[] = [...start, selection,
    { ts: 111, type: 'attempt', candidate: 0, seed: 0, code_hash: 'a2', measured: 0.7 },
    { ts: 112, type: 'repo_diff', result: 'kept', from_hash: 'a', to_hash: 'a2' },
  ]
  assert.equal(build(repaired).stats!.mean, 0.827)
  repaired.push({ ts: 113, type: 'attempt', candidate: 0, seed: 0, code_hash: 'a3', measured: 0.4 },
    { ts: 114, type: 'repo_diff', result: 'adopted', from_hash: 'a', to_hash: 'a3' })
  assert.equal(build(repaired).stats!.mean, 0.4)
})

test('selected seeds count once each; other candidates never overwrite a selected measurement', () => {
  const events: Event[] = [...start, selection,
    { ts: 111, type: 'attempt', candidate: 0, seed: 1, code_hash: 'a', measured: 0.8 },
    { ts: 112, type: 'attempt', candidate: 0, seed: 2, code_hash: 'a', measured: 0.85 },
    { ts: 113, type: 'attempt', candidate: 1, seed: 0, code_hash: 'b2', measured: 0.1 },
  ]
  assert.deepEqual(build(events).attempts.map(a => a.measured), [0.827, 0.8, 0.85])
})

test('legacy single-code events still show measurements without candidate selection', () => {
  const m = build([{ ts: 100, type: 'attempt', seed: 0, code_hash: 'legacy', measured: 0.5 }])
  assert.equal(m.candidateSelectionPending, false)
  assert.equal(m.stats!.mean, 0.5)
})

test('final measurement selection preserves accepted mean repair and never promotes rejected attempts', () => {
  const events: Event[] = [...start, selection,
    { ts: 111, type: 'attempt', candidate: 0, seed: 0, code_hash: 'new', measured: 0.4, seconds: 60 },
    { ts: 112, type: 'measurements_selected', attempts: [{ seed: 0, code_hash: 'new', measured: 0.4 }] },
  ]
  assert.deepEqual(build(events).attempts, [{ seed: 0, hash: 'new', measured: 0.4, seconds: 60 }])
  events.push({ ts: 113, type: 'measurements_selected', attempts: [] })
  assert.equal(build(events).stats, null)
})

test('older mean-repair logs recover only a complete code group agreeing with their final verdict', () => {
  const events: Event[] = [...start, selection,
    ...[0, 1, 2].map(seed => ({ ts: 111 + seed, type: 'attempt', candidate: 0, seed, code_hash: 'new', measured: 0.5 })),
    { ts: 115, type: 'verdict', mean: 0.5, std: 0 },
  ]
  assert.deepEqual(build(events).attempts.map(a => a.hash), ['new', 'new', 'new'])
  events[events.length - 1].mean = 0.6
  assert.equal(build(events).measurementSelectionUnavailable, true)
  assert.equal(build(events).stats, null)
  events[events.length - 1].mean = 0.5
  events[events.length - 1].std = 0.1
  assert.equal(build(events).measurementSelectionUnavailable, true)
})

test('a mean repair request does not falsely claim failure, and acceptance changes selected code', () => {
  const events: Event[] = [...start, selection,
    { ts: 111, type: 'fault_repair', runde: 'mittelwert', from_hash: 'a', measured: 0.827 },
    { ts: 112, type: 'attempt', candidate: 0, seed: 0, code_hash: 'new', measured: 0.5 },
  ]
  assert.ok(build(events).phases.flatMap(p => p.rows).some(r => r.title.includes('a code repair is requested')))
  events.push({ ts: 113, type: 'fault_repair_result', result: 'adopted', from_hash: 'a', to_hash: 'new' })
  assert.equal(build(events).stats!.mean, 0.5)
})

test('identical code shared by candidates keeps attempt labels and the chosen measurement separate', () => {
  const events: Event[] = [
    { ts: 100, type: 'code_written', candidate: 0, code_hash: 'shared' },
    { ts: 101, type: 'code_written', candidate: 1, code_hash: 'shared' },
    { ts: 102, type: 'attempt', candidate: 0, seed: 0, code_hash: 'shared', measured: 0.4 },
    { ts: 103, type: 'attempt', candidate: 1, seed: 0, code_hash: 'shared', measured: 0.9 },
    { ts: 104, type: 'fault_repair', result: 'repariert', from_hash: 'shared', to_hash: 'other' },
    { ts: 105, type: 'candidates', chosen: 0, chosen_hash: 'shared' },
  ]
  const m=build(events)
  assert.equal(m.stats!.mean,0.4)
  const rows=m.phases.flatMap(p=>p.rows)
  assert.ok(rows.some(r=>r.title.startsWith('Candidate 1 · Seed 0') && r.raw[0]?.measured===0.4))
  assert.ok(rows.some(r=>r.title.startsWith('Candidate 2 · Seed 0') && r.raw[0]?.measured===0.9))
})

test('validation repairs are described separately from historical test-score repairs', () => {
  const m = build([...start, { ts: 110, type: 'fault_repair', basis: 'validation', result: 'repariert' }])
  const rows = m.phases.flatMap(p => p.rows)
  assert.ok(rows.some(r => r.title === 'Validation check failed; the implementation was repaired' && r.detail?.includes('Only validation data')))
  assert.ok(!rows.some(r => r.title.includes('Baseline check failed after all seeds')))
})

test('completed event charts agree with published verdict means', () => {
  const root = new URL('../public/data/', import.meta.url)
  for (const paper of readdirSync(root, { withFileTypes: true }).filter(x => x.isDirectory())) {
    const dir = new URL(`${paper.name}/`, root)
    for (const run of readdirSync(dir, { withFileTypes: true }).filter(x => x.isDirectory())) {
      const path = new URL(`${run.name}/events.jsonl`, dir)
      let events: Event[]
      try { events = readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line)) }
      catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue; throw error }
      const m = buildRunModel(events, { running: false })
      if (m.verdict && m.stats) assert.ok(Math.abs(m.stats.mean - m.verdict.mean) < 1e-9, path.pathname)
    }
  }
})

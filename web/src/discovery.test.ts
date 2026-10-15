import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { deriveView, tickedRuntime, parseLiveEvent, savedCheckForClaim, humanReject, type DEvent } from './discovery.ts'

test('unverified horizons in new and historic searches are extraction limits, not claims about the paper', () => {
  for (const reason of ['Horizont nicht 96', 'Prediction horizon could not be verified from the extracted claim']) {
    assert.match(humanReject('number', reason), /extraction could not confirm/)
    assert.match(humanReject('number', reason), /does not establish/)
  }
})

test('a cited baseline is explained as a claim selection problem', () => {
  assert.match(humanReject('quote', 'Selected method is cited prior work, not a method proposed in this paper'), /earlier model cited/)
})

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


const catalog = JSON.parse(readFileSync(new URL('../public/data/catalog.json', import.meta.url), 'utf8'))
const recorded = readFileSync(new URL('../public/data/discovery/replay.jsonl', import.meta.url), 'utf8').trim().split('\n').map((line) => JSON.parse(line))
const discovered = deriveView(recorded).papers.filter((paper) => paper.state === 'candidate')

test('recorded candidates link to matching saved checks with their actual negative verdicts', () => {
  const entries = discovered.map((paper) => savedCheckForClaim(paper, catalog))
  assert.deepEqual(entries.map((entry) => entry?.arxiv_id), ['2511.08094', '2601.21866'])
  assert.deepEqual(entries.map((entry) => entry?.verdict?.urteil), ['nicht reproduziert', 'Agenten-Fehler vermutet'])
})

test('a saved result from another dataset, method, target or horizon is not assigned to the discovered claim', () => {
  const graph = discovered.find((paper) => paper.id === '2511.08094')!
  const series = discovered.find((paper) => paper.id === '2601.21866')!
  const graphEntry = savedCheckForClaim(graph, catalog)!
  const seriesEntry = savedCheckForClaim(series, catalog)!
  assert.equal(savedCheckForClaim(graph, [{ ...graphEntry, task: 'Cora' }]), undefined)
  for (const claim of [{ ...graphEntry.claim, method: 'GCN' }, { ...graphEntry.claim, target: 80 }]) {
    assert.equal(savedCheckForClaim(graph, [{ ...graphEntry, claim }]), undefined)
  }
  assert.equal(savedCheckForClaim(series, [{ ...seriesEntry, claim: { ...seriesEntry.claim, horizon: 192 } }]), undefined)
})

test('accuracy units normalize consistently and the newest matching check wins regardless of verdict', () => {
  const paper = discovered.find((p) => p.id === '2511.08094')!
  const saved = savedCheckForClaim(paper, catalog)!
  const older = { ...saved, checked_at: '2026-09-30T00:00:00Z', verdict: { ...saved.verdict, urteil: 'reproduziert' } }
  const newest = { ...saved, claim: { ...saved.claim, target: 0.8292, metric: 'acc' } }
  assert.equal(savedCheckForClaim(paper, [older, newest]), newest)
  assert.equal(savedCheckForClaim({ ...paper, claim: undefined }, catalog), undefined)
})

test('live candidate cards retain the exact server ID, claim and eligibility reason', () => {
  const base = { type: 'candidate', ts: 10, arxiv_id: '2412.03864', title: 'SimMLP', dataset: 'Cora' }
  const view = deriveView([
    { ...base, type: 'searching', claim: { text: JSON.stringify({ method: 'old method', target: 99 }) } },
    { ...base, catalog_id: 'disc_2412_03864_claimA', runnable: false, run_reason: 'The Cora split is not established.',
      claim: { method: 'SimMLP', metric: 'accuracy', target: 81.4 } },
  ])
  assert.equal(view.papers[0].catalogId, 'disc_2412_03864_claimA')
  assert.equal(view.papers[0].runnable, false)
  assert.equal(view.papers[0].runReason, 'The Cora split is not established.')
  assert.equal(view.papers[0].claim?.target, 81.4)
})

test('different saved claims from the same paper retain separate start IDs and targets', () => {
  const base = { type: 'candidate', ts: 10, arxiv_id: '2505.20716', title: 'Paper', dataset: 'ETTh1', runnable: true }
  const view = deriveView([
    { ...base, catalog_id: 'disc_2505_20716_A', claim: { method: 'A', metric: 'mse', target: .378 } },
    { ...base, catalog_id: 'disc_2505_20716_B', claim: { method: 'B', metric: 'mse', target: .366 } },
  ])
  assert.deepEqual(view.papers.map((paper) => paper.catalogId).sort(), ['disc_2505_20716_A', 'disc_2505_20716_B'])
  assert.deepEqual(view.papers.map((paper) => paper.claim?.target).sort(), [.366, .378])
})

test('a found claim shows its readiness check and the saved result of a finished one', () => {
  const found = { type: 'candidate', ts: 1, arxiv_id: '2410.09348', catalog_id: 'c1', title: 'BANGS', runnable: true } as DEvent
  const checking = deriveView([found, { type: 'readiness', ts: 2, catalog_id: 'c1', state: 'checking' } as DEvent])
  assert.equal(checking.papers.find((p) => p.catalogId === 'c1')?.readiness?.state, 'checking')
  const blocked = deriveView([found, { type: 'readiness', ts: 2, catalog_id: 'c1', state: 'checking' } as DEvent,
    { type: 'readiness', ts: 3, catalog_id: 'c1', state: 'blocked', stage: "authors' code", reason: 'no repository' } as DEvent])
  assert.deepEqual(blocked.papers.find((p) => p.catalogId === 'c1')?.readiness, { state: 'blocked', reason: 'no repository', stage: "authors' code" })
  const saved = deriveView([{ ...found, readiness: { ready: true, stage: 'ready', reason: null } } as DEvent])
  assert.equal(saved.papers.find((p) => p.catalogId === 'c1')?.readiness?.state, 'ready')
})

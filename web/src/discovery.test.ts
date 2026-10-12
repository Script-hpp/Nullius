import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { deriveView, tickedRuntime, parseLiveEvent, savedCheckForClaim, type DEvent } from './discovery.ts'

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

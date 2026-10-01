import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { FILM_MS, addressBar, branchX, branches, claimEvidence, codeLines, diffFlash, dossierAt, durationText, measurements, planBullets, presentScenes, repairScene, sceneAt, scenePlan, sourceDomains, tavilySentence, typed, type SceneId } from './film.ts'
import type { Event } from './data.ts'

const all: SceneId[] = ['tavily', 'code', 'repair', 'sandbox', 'harness', 'verdict']
const events = JSON.parse(readFileSync(new URL('./fixtures/nlinear_story_events.json', import.meta.url), 'utf8')) as Event[]

test('plan: always 30 s, scenes follow each other without gaps, also without the repair scene', () => {
  for (const ids of [all, all.filter((s) => s !== 'repair'), ['harness', 'verdict'] as SceneId[]]) {
    const plan = scenePlan(new Set(ids))
    assert.equal(plan[0].start, 0)
    assert.equal(plan[plan.length - 1].end, FILM_MS)
    plan.slice(1).forEach((s, i) => assert.equal(s.start, plan[i].end))
    assert.deepEqual(plan.map((s) => s.id), ids)
  }
})

test('plan: the repair scene shortens nothing it does not have to, the others stretch to fill 30 s', () => {
  const with_ = scenePlan(new Set(all)); const without = scenePlan(new Set(all.filter((s) => s !== 'repair')))
  assert.ok(without[0].end - without[0].start > with_[0].end - with_[0].start)
})

test('sceneAt: index and progress, clamped at both ends', () => {
  const plan = scenePlan(new Set(all))
  assert.deepEqual(sceneAt(plan, 0), { index: 0, p: 0 })
  assert.equal(sceneAt(plan, FILM_MS + 99).index, plan.length - 1)
  assert.equal(sceneAt(plan, FILM_MS).p, 1)
  const mid = (plan[2].start + plan[2].end) / 2
  assert.deepEqual([sceneAt(plan, mid).index, Math.round(sceneAt(plan, mid).p * 100)], [2, 50])
  assert.deepEqual(sceneAt([], 5), { index: 0, p: 0 })
})

test('claim evidence: the real quote, table and page; nothing without a quote', () => {
  const c = { target_evidence: { quote: 'GCN (this paper) 70 .3 (7s) 81 .5 (4s)', table: 'Table 2', url: 'https://arxiv.org/pdf/1609.02907', arxiv_version: 'v4' } }
  assert.deepEqual(claimEvidence(c), { quote: 'GCN (this paper) 70 .3 (7s) 81 .5 (4s)', table: 'Table 2', url: 'https://arxiv.org/pdf/1609.02907', version: 'v4' })
  assert.equal(claimEvidence({ target_evidence: null }), null)
  assert.equal(addressBar('https://www.arxiv.org/pdf/1609.02907'), 'arxiv.org/pdf/1609.02907')
})

test('source domains: the evidence page first, then Tavily targets, each once', () => {
  const ev = [{ ts: 0, type: 'tavily_call', target: 'https://github.com/tkipf/gcn' }, { ts: 1, type: 'tavily_call', target: 'https://arxiv.org/pdf/1609.02907' }, { ts: 2, type: 'tavily_call', target: 'not a url' }] as Event[]
  assert.deepEqual(sourceDomains(ev, 'https://arxiv.org/pdf/1609.02907'), ['arxiv.org', 'github.com'])
  assert.deepEqual(sourceDomains([], undefined), [])
})

test('code lines: no blanks or comment-only lines, limited, long lines cut', () => {
  const code = '# plan comment\n\nimport numpy as np\n\nx = 1  # note\n' + 'y = 2\n'.repeat(30) + 'z = "' + 'a'.repeat(200) + '"\n'
  const l = codeLines(code, 5)
  assert.equal(l.length, 5)
  assert.equal(l[0], 'import numpy as np')
  assert.ok(!l.some((x) => x.startsWith('#')))
  assert.ok(codeLines('z = "' + 'a'.repeat(200) + '"')[0].length <= 78)
})

test('plan bullets: markdown stripped, short lines dropped', () => {
  const b = planBullets('**Reproduction Plan: GCN on Cora**\n\n1. **Model**: 2-layer GCN with dropout 0.5\n- Optimizer: Adam, lr 0.01\nok\n')
  assert.deepEqual(b, ['Reproduction Plan: GCN on Cora', 'Model: 2-layer GCN with dropout 0.5', 'Optimizer: Adam, lr 0.01'])
})

test('typed: a prefix of the text, clamped', () => {
  assert.equal(typed('abcdef', 0.5), 'abc')
  assert.equal(typed('abcdef', 2), 'abcdef')
  assert.equal(typed('abcdef', -1), '')
})

test('branches: one per seed, a valid measurement wins over an earlier failure; measurements keep event order', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 1, measured: null }, { ts: 1, type: 'attempt', claim_id: 'c', seed: 0, measured: 0.8 },
    { ts: 2, type: 'attempt', claim_id: 'c', seed: 1, measured: 0.7 }, { ts: 3, type: 'attempt', claim_id: 'c', seed: 2, measured: null },
  ] as Event[]
  assert.deepEqual(branches(ev), [{ seed: 0, ok: true }, { seed: 1, ok: true }, { seed: 2, ok: false }])
  assert.deepEqual(measurements(ev, 'c'), [0.8, 0.7])
  assert.deepEqual([branchX(0, 1), branchX(0, 3), branchX(2, 3)], [0.5, 0.06, 0.94])
})

test('dossierAt: the first k measurements, the verdict only when asked', () => {
  const ev = [0.8, 0.82, 0.81].map((m, i) => ({ ts: i, type: 'attempt', claim_id: 'c', seed: i, measured: m })) as Event[]
  const d = { claims: [{ id: 'c' }], attempts: [], verdicts: [{ claim_id: 'c' }] }
  assert.equal(dossierAt(d, 'c', ev, 2, false).attempts.length, 2)
  assert.deepEqual(dossierAt(d, 'c', ev, 2, false).verdicts, [])
  assert.equal(dossierAt(d, 'c', ev, 3, true).verdicts.length, 1)
})

test('repair scene on a real run: first repair card, a sentence that is true for it, the version counter', () => {
  const r = repairScene(events, 'nlinear_etth1_96_mse')!
  assert.equal(r.card.source, 'error')
  assert.equal(r.sentence, 'The code failed. The agent reads the error, searches the web and fixes it.')
  assert.deepEqual(r.version, { from: 1, to: 2 })
  assert.equal(repairScene(events, 'other'), null)
})

test('repair sentence without a search does not mention one', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 0, n: 1, code_hash: 'a', status: 'fehler', measured: null, error: 'ValueError: x' },
    { ts: 1, type: 'repair', claim_id: 'c', seed: 0, from_hash: 'a', error: 'ValueError: x' },
  ] as Event[]
  assert.equal(repairScene(ev, 'c')!.sentence, 'The code failed. The agent reads the error and fixes it.')
})

test('diff flash: only changed lines, limited', () => {
  const f = diffFlash('a\nb\nc\nd', 'a\nB\nc\nd\ne\nf\ng', 4)
  assert.deepEqual(f.map((l) => l.t), ['del', 'add', 'add', 'add'])
  assert.deepEqual(diffFlash('x', 'x'), [])
})

test('present scenes: a scene appears only if its data exists', () => {
  const claim = { id: 'c', target_evidence: null }
  const none = presentScenes({ events: [], claim, hasPlan: false, hasCode: false, hasVerdict: false, repair: false })
  assert.equal(none.size, 0)
  const ev = [{ ts: 0, type: 'tavily_call', target: 'https://x.org/a' }, { ts: 1, type: 'attempt', claim_id: 'c', seed: 0, measured: 0.8 }] as Event[]
  const some = presentScenes({ events: ev, claim, hasPlan: true, hasCode: false, hasVerdict: true, repair: false })
  assert.deepEqual([...some].sort(), ['code', 'harness', 'sandbox', 'tavily', 'verdict'])
})

test('duration text', () => {
  const e = (t: number) => ({ ts: t, type: 'x' }) as Event
  assert.equal(durationText([e(0), e(41 * 60)]), '41 minutes')
  assert.equal(durationText([e(0)]), null)
})

test('first sentence only claims a found number when there is evidence', () => {
  assert.match(tavilySentence(true), /finds the number it reports/)
  assert.doesNotMatch(tavilySentence(false), /number/)
})

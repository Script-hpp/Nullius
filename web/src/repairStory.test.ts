import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildStory, domainOf, lineDiff, problemOf, type Card } from './repairStory.ts'

// Real events of a run (zeng_nlinear_2023, 3 candidates in parallel, 5 error repairs, 3 fault repairs), trimmed to the fields the story reads.
const events = JSON.parse(readFileSync(new URL('./fixtures/nlinear_story_events.json', import.meta.url), 'utf8'))
const story = buildStory(events, 'nlinear_etth1_96_mse')
const repairs = story.filter((c): c is Extract<Card, { kind: 'repair' }> => c.kind === 'repair')

test('incomplete repository reviews are explicit, including historical no_deviation events with cuts', () => {
  for (const event of [
    { type: 'repo_diff', ts: 1, claim_id: 'c', result: 'incomplete' },
    { type: 'repo_diff', ts: 1, claim_id: 'c', result: 'no_deviation', truncation: [{ file: 'layers/Retrieval.py', kept_chars: 0 }] },
    { type: 'repo_diff', ts: 1, claim_id: 'c', result: 'no_deviation', dropped: 1 },
  ]) {
    const card = buildStory([event], 'c').find((c) => c.kind === 'repo_diff')!
    assert.match(card.text, /comparison was incomplete/)
    assert.match(card.text, /Method fidelity remains unchecked/)
  }
})

test('cards: one per repair event, plus the candidate choice and the repo comparison', () => {
  assert.equal(repairs.filter((r) => r.source === 'error').length, 5)
  assert.equal(repairs.filter((r) => r.source === 'fault').length, 3)
  assert.equal(story.filter((c) => c.kind === 'candidates').length, 1)
  assert.equal(story.filter((c) => c.kind === 'repo_diff').length, 1)
  assert.match((story.find((c) => c.kind === 'candidates') as { text: string }).text, /3 versions .* best score on validation data/)
})

test('each error repair carries the error line and the search that came right before it', () => {
  const first = repairs.find((r) => r.source === 'error')!
  assert.equal(first.fromHash, 'e33c912af040')
  assert.match(first.problem, /RuntimeError|mat1|shape|size|Error/)
  assert.equal(first.search?.query, 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (1xN and NxN)')
  assert.ok(first.search!.links.length >= 1 && first.search!.links[0].domain === 'discuss.pytorch.org')
  // the loop searched 4 times for 5 error repairs (the fifth was a smoke-test crash without a search): the card says nothing then
  assert.equal(repairs.filter((r) => r.source === 'error' && r.search !== null).length, 4)
})

test('parallel candidates: results are matched in order, never to the same hash twice', () => {
  const errs = repairs.filter((r) => r.source === 'error')
  const targets = errs.map((r) => r.toHash).filter(Boolean)
  assert.equal(new Set(targets).size, targets.length)
  assert.equal(errs[0].toHash, '7f1cece332d0')
  assert.equal(errs[1].toHash, '4696c7a0bc01')
  assert.ok(errs[0].toVersion === 2 && errs[0].fromVersion === 1)
})

test('fault repairs name both hashes and the measured value; the outcome is the first attempt of the new code', () => {
  const f = repairs.find((r) => r.source === 'fault')!
  assert.equal(f.fromHash, '614354c70832')
  assert.equal(f.toHash, '0f6dc69f2f85')
  assert.match(f.problem, /It measured 1\.300/)
  assert.deepEqual(f.outcome, { tone: 'ok', text: 'It ran. Valid measurement: 0.526.' })
})

test('a failed new version says so with the last error line', () => {
  const failing = repairs.filter((r) => r.outcome.tone === 'fail')
  assert.ok(failing.length >= 1)
  assert.ok(failing.every((r) => r.outcome.text.startsWith('It failed again:') || r.outcome.text.startsWith('It was rejected:')))
})

test('parallel candidates are told apart: each card names its candidate and results follow the same candidate', () => {
  const errs = repairs.filter((r) => r.source === 'error')
  assert.deepEqual(errs.map((r) => r.candidate), [2, 1, 2, 1, 1])
  // candidate 2: e33c -> 7f1c (n = 2); candidate 1: 46a2 -> 4696 (n = 2) -> 1db7 (n = 3)
  assert.deepEqual(errs.map((r) => [r.fromHash, r.toHash]).slice(0, 4), [
    ['e33c912af040', '7f1cece332d0'], ['46a26ac1502a', '4696c7a0bc01'], ['7f1cece332d0', '0ed5e001fe19'], ['4696c7a0bc01', '1db723ca1231'],
  ])
  assert.deepEqual(errs.map((r) => [r.fromVersion, r.toVersion]), [[1, 2], [1, 2], [2, 3], [2, 3], [3, 4]])
})

test('to_hash in the repair event wins over the inferred result', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 0, n: 1, code_hash: 'aaa', status: 'fehler', measured: null, error: 'ValueError: x' },
    { ts: 1, type: 'repair', claim_id: 'c', seed: 0, from_hash: 'aaa', to_hash: 'bbb', error: 'ValueError: x' },
    { ts: 2, type: 'attempt', claim_id: 'c', seed: 0, n: 2, code_hash: 'zzz', status: 'fehler', measured: null, error: 'other' },
    { ts: 3, type: 'attempt', claim_id: 'c', seed: 0, n: 2, code_hash: 'bbb', status: 'stabil', measured: 0.5, error: null },
  ] as never
  const [card] = buildStory(ev, 'c') as Extract<Card, { kind: 'repair' }>[]
  assert.equal(card.toHash, 'bbb')
  assert.deepEqual(card.outcome, { tone: 'ok', text: 'It ran. Valid measurement: 0.500.' })
})

test('a repair_result event after the repair names the fixed code', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 0, n: 1, code_hash: 'aaa', status: 'fehler', measured: null, error: 'ValueError: x' },
    { ts: 1, type: 'repair', claim_id: 'c', seed: 0, from_hash: 'aaa', error: 'ValueError: x' },
    { ts: 2, type: 'attempt', claim_id: 'c', seed: 0, n: 2, code_hash: 'zzz', status: 'fehler', measured: null, error: 'other' },
    { ts: 3, type: 'repair_result', claim_id: 'c', seed: 0, from_hash: 'aaa', to_hash: 'bbb' },
    { ts: 4, type: 'attempt', claim_id: 'c', seed: 0, n: 2, code_hash: 'bbb', status: 'stabil', measured: 0.5, error: null },
  ] as never
  assert.equal((buildStory(ev, 'c')[0] as Extract<Card, { kind: 'repair' }>).toHash, 'bbb')
})

test('events of other claims are ignored', () => {
  assert.deepEqual(buildStory(events, 'some_other_claim'), [])
})

test('problemOf translates the loop’s German static-check text and picks the exception line', () => {
  assert.deepEqual(problemOf("Statische Pruefung: URL im Code: '://' in Zeile 83"), { title: 'The static code check rejected the code', line: "URL in code: '://' in line 83" })
  assert.equal(problemOf('Traceback (most recent call last):\n  File "x", line 1\nNameError: name \'horizon\' is not defined').line, "NameError: name 'horizon' is not defined")
  assert.equal(problemOf('Smoke run (SMOKE=1) crashed:\nTraceback\nValueError: bad').title, 'The quick form test crashed')
})

test('domainOf drops www and survives garbage', () => {
  assert.equal(domainOf('https://www.stackoverflow.com/questions/1'), 'stackoverflow.com')
  assert.equal(domainOf('not a url'), 'not a url')
})

test('lineDiff: only changed lines with two lines of context, in hunks', () => {
  const a = Array.from({ length: 20 }, (_, i) => `line ${i}`).join('\n')
  const b = a.replace('line 3', 'LINE 3').replace('line 15', 'LINE 15\nextra')
  const h = lineDiff(a, b)
  assert.equal(h.length, 2)
  assert.deepEqual(h[0].map((l) => l.t), ['ctx', 'ctx', 'del', 'add', 'ctx', 'ctx'])
  assert.equal(h[0].filter((l) => l.t === 'del').length, 1)
  assert.equal(h[1].filter((l) => l.t === 'add').length, 2)
  assert.deepEqual(lineDiff(a, a), [])
})

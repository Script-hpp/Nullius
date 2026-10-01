import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildStory, domainOf, lineDiff, problemOf, type Card } from './repairStory.ts'

// Real events of a run (zeng_nlinear_2023, 3 candidates in parallel, 5 error repairs, 3 fault repairs), trimmed to the fields the story reads.
const events = JSON.parse(readFileSync(new URL('./fixtures/nlinear_story_events.json', import.meta.url), 'utf8'))
const story = buildStory(events, 'nlinear_etth1_96_mse')
const repairs = story.filter((c): c is Extract<Card, { kind: 'repair' }> => c.kind === 'repair')

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
  assert.equal(errs[0].toHash, '7f1cece332d0')          // first repair of e33c... -> first attempt with n = 2
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

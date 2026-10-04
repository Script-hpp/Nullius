import test from 'node:test'
import assert from 'node:assert/strict'
import { referenceSentence } from './referenceBench.ts'

const b = { version: 'v4', arm: 'with Tavily', reached: 3, runs: 3, units_matched: 2, units: 8 }

test('sentence uses the exported figures', () => {
  assert.equal(referenceSentence(b),
    'In benchmark v4 (with Tavily), 3 of 3 fresh runs reached the target. Across all 8 paper-claim units, 2 matched the reference verdict.')
  assert.match(referenceSentence({ ...b, reached: 2 }), /2 of 3 fresh runs/)
})

test('singular wording for one run or one unit', () => {
  assert.match(referenceSentence({ ...b, reached: 1, runs: 1 }), /1 of 1 fresh run reached/)
  assert.match(referenceSentence({ ...b, units: 1, units_matched: 1 }), /all 1 paper-claim unit, 1 matched/)
})

test('missing or empty figures give no numbers', () => {
  for (const x of [null, undefined, { ...b, runs: 0 }]) assert.doesNotMatch(referenceSentence(x), /\d/)
})

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { describeDiff } from './delta.ts'

test('inside half of the last shown digit the value matches the claim', () => {
  assert.equal(describeDiff(0.00003, false).text, 'matches the claim')
  assert.equal(describeDiff(-0.0004, false).match, true)
  assert.equal(describeDiff(0.04, true).text, 'matches the claim')
})

test('outside it the signed difference is shown', () => {
  assert.equal(describeDiff(0.003, false).text, '+0.003 vs. claim')
  assert.equal(describeDiff(-0.0123, false).text, '−0.012 vs. claim')
  assert.equal(describeDiff(-1.24, true).text, '−1.2 pp vs. claim')
  assert.equal(describeDiff(0.0005, false).match, false)
  assert.equal(describeDiff(0.0005, false).sign, 1)
})

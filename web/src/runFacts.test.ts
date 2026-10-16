import { test } from 'node:test'
import assert from 'node:assert/strict'
import { runFacts } from './runFacts.ts'

test('facts line from the run', () => {
  assert.equal(runFacts({ seeds: 3, seconds: 840, costUsd: 0.3101 }), '3 seeds · 14 min · $0.31')
  assert.equal(runFacts({ seeds: 1, seconds: 45, costUsd: null }), '1 seed · 45 s')
  assert.equal(runFacts({ seeds: 0, seconds: null, costUsd: undefined }), '')
})

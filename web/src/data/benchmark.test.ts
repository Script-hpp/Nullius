import test from 'node:test'
import assert from 'node:assert/strict'
import { UNITS, count, FIGURES, HELD_OUT } from './benchmark.ts'

test('unit flags add up to the published counts', () => {
  assert.equal(UNITS.length, 8)
  assert.equal(count('paperOnly'), FIGURES.paperOnly.matched)
  assert.equal(count('withCode'), FIGURES.withCode.matched)
})

test('held-out and placebo figures', () => {
  assert.equal(HELD_OUT.matched, 0)
  assert.deepEqual(FIGURES.placebo.cg, [0, 9])
  assert.deepEqual(FIGURES.placebo.v4, [0, 48])
})

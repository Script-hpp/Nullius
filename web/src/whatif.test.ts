import test from 'node:test'
import assert from 'node:assert/strict'
import { fromUnits, sliderRange, toUnits, verdictFor, verdictForSlider, whatIfEligible } from './whatif.ts'

// Binary-exact numbers, so that "exactly on the tolerance" is really exact.
const mean = 0.5
const tol = 0.25

test('exactly on the tolerance is reproduced, just above is partial (both directions)', () => {
  assert.equal(verdictFor(mean, mean + tol, tol), 'reproduziert')
  assert.equal(verdictFor(mean, mean - tol, tol), 'reproduziert')
  assert.equal(verdictFor(mean, mean + tol + 1e-9, tol), 'teilweise')
  assert.equal(verdictFor(mean, mean - tol - 1e-9, tol), 'teilweise')
})

test('exactly three tolerances is partial, just above is not reproduced (both directions)', () => {
  assert.equal(verdictFor(mean, mean + 3 * tol, tol), 'teilweise')
  assert.equal(verdictFor(mean, mean - 3 * tol, tol), 'teilweise')
  assert.equal(verdictFor(mean, mean + 3 * tol + 1e-9, tol), 'nicht reproduziert')
  assert.equal(verdictFor(mean, mean - 3 * tol - 1e-9, tol), 'nicht reproduziert')
})

test('no distance, a tiny distance and a huge distance', () => {
  assert.equal(verdictFor(mean, mean, tol), 'reproduziert')
  assert.equal(verdictFor(mean, mean + 100 * tol, tol), 'nicht reproduziert')
})

test('same rule for a loss metric: only the distance counts, not the direction', () => {
  // MSE: mean 0.375, tolerance 0.01 (the DLinear claim); a claim below or above the mean is judged alike
  assert.equal(verdictFor(0.3809, 0.375, 0.01), 'reproduziert')
  assert.equal(verdictFor(0.3809, 0.39, 0.01), 'reproduziert')
  assert.equal(verdictFor(0.3809, 0.36, 0.01), 'teilweise')
  assert.equal(verdictFor(0.3809, 0.30, 0.01), 'nicht reproduziert')
})

test('the slider grid: ties on 1 and 3 tolerances stay inside despite floating point error', () => {
  const m = 0.821, t = 0.01
  for (const [u, want] of [[-1, 'reproduziert'], [1, 'reproduziert'], [-3, 'teilweise'], [3, 'teilweise'], [1.1, 'teilweise'], [3.1, 'nicht reproduziert'], [0, 'reproduziert']] as const) {
    assert.equal(verdictForSlider(m, fromUnits(m, u, t), t), want, `u=${u}`)
  }
  // all grid points of the range: the verdict follows |u| exactly
  for (let k = -60; k <= 60; k++) {
    const u = k / 10
    const want = Math.abs(k) <= 10 ? 'reproduziert' : Math.abs(k) <= 30 ? 'teilweise' : 'nicht reproduziert'
    assert.equal(verdictForSlider(m, fromUnits(m, u, t), t), want, `k=${k}`)
  }
})

test('units round trip and the range covers the real claim', () => {
  assert.ok(Math.abs(toUnits(0.821, fromUnits(0.821, 2.5, 0.01), 0.01) - 2.5) < 1e-9)
  assert.deepEqual(sliderRange(0.2), { min: -6, max: 6 })
  assert.ok(sliderRange(27.7).max >= 28 && sliderRange(-9.3).min <= -10)
})

test('the card is only for verdicts that rest on measurements and distance', () => {
  const v = (urteil: string, m: unknown = 0.8) => ({ urteil, measured_mean: m })
  assert.equal(whatIfEligible(v('reproduziert'), 5, 0.01), true)
  assert.equal(whatIfEligible(v('teilweise'), 5, 0.01), true)
  assert.equal(whatIfEligible(v('nicht reproduziert'), 5, 0.01), true)
  assert.equal(whatIfEligible(v('nicht reproduzierbar (Setup)', NaN), 0, 0.01), false)
  assert.equal(whatIfEligible(v('Agenten-Fehler vermutet'), 5, 0.01), false)
  assert.equal(whatIfEligible(v('reproduziert'), 0, 0.01), false)
  assert.equal(whatIfEligible(v('reproduziert'), 5, 0), false)
  assert.equal(whatIfEligible(undefined, 5, 0.01), false)
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { countAt, easeOut } from './countUp.ts'

test('starts at 0, ends exactly at the target', () => {
  assert.equal(countAt(9, 0), 0)
  assert.equal(countAt(9, 700), 9)
  assert.equal(countAt(9, 5000), 9)
})

test('ease-out: more than half done at half the time, never above target, never decreasing', () => {
  assert.ok(easeOut(0.5) > 0.5)
  let prev = 0
  for (let t = 0; t <= 700; t += 25) {
    const v = countAt(12, t)
    assert.ok(v >= prev && v <= 12)
    prev = v
  }
})

test('zero duration shows the target directly', () => {
  assert.equal(countAt(4, 0, 0), 4)
})

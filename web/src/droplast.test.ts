import test from 'node:test'
import assert from 'node:assert/strict'
import { droplastParagraphs, type DropLastNote } from './droplast.ts'

const base: DropLastNote = {
  metric: 'mse', claimed: 0.366, tolerance: 0.01, full: 0.3704, droplast: 0.3665, batch_size: 128, windows_dropped: 97, windows_total: 2785,
  kind: 'within_tolerance', batch_context: 'the batch size of the PatchTST and RLinear ETTh1 scripts', sources: [],
}

test('within tolerance: says the full value is still within tolerance', () => {
  const t = droplastParagraphs(base).join(' ')
  assert.match(t, /last 97 of 2785 test windows are skipped/)
  assert.match(t, /still within tolerance of the claimed 0\.3660/)
  assert.doesNotMatch(t, /may explain/)
})

test('outside tolerance: says it may explain the difference', () => {
  const t = droplastParagraphs({ ...base, kind: 'outside_tolerance', claimed: 0.35, full: 0.3704, droplast: 0.351 }).join(' ')
  assert.match(t, /may explain the difference/)
  assert.doesNotMatch(t, /within tolerance/)
})

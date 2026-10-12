import test from 'node:test'
import assert from 'node:assert/strict'
import { formatMetric, formatSpread } from './metricFormat.ts'

test('TSMixer MSE and its nonzero spread remain decimals; accuracy remains a percentage', () => {
  assert.equal(formatMetric('mse', 0.361), '0.3610')
  assert.equal(formatMetric('mae', 0.392), '0.3920')
  assert.equal(formatSpread('mse', 0.0002884589900119613), '0.0003')
  assert.notEqual(Number(formatSpread('mse', 0.00000001)), 0)
  assert.equal(formatMetric('accuracy', 0.815), '81.5%')
  assert.equal(formatMetric('accuracy', 0), '0.0%')
  assert.equal(formatSpread('accuracy', 0.008), '0.8')
  for (const missing of [undefined, null, NaN, Infinity]) assert.equal(formatMetric('mse', missing), '–')
})

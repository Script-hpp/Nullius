import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cardClaim, noVerdictLabel } from './catalogCard.ts'

const base = { title: 'FITS: Modeling Time Series with 10k Parameters', dataset: 'ETTh1', task: 'ETTh1' }

test('claim in plain words', () => {
  assert.equal(cardClaim({ ...base, claim: { method: 'FITS', metric: 'mse', target: 0.372 } }), 'FITS: forecast error 0.372 on ETTh1')
  assert.equal(cardClaim({ ...base, dataset: 'Cora', claim: { method: 'GCN (this paper)', metric: 'accuracy', target: 0.815 } }), 'GCN: accuracy 81.5% on Cora')
  assert.equal(cardClaim({ ...base, claim: { method: 'X', metric: 'mae', target: 0.4 } }), 'X: forecast error (MAE) 0.4 on ETTh1')
  assert.equal(cardClaim({ ...base, claim: { metric: 'mse', target: 0.4 } }), 'FITS: forecast error 0.4 on ETTh1')
})

test('labels for entries without a verdict', () => {
  assert.equal(noVerdictLabel('ungeprueft', false), 'Not checked yet')
  assert.equal(noVerdictLabel('ungeprueft', true), 'Last check stopped')
  assert.equal(noVerdictLabel('lauf_fehlgeschlagen', false), 'Run failed')
  assert.equal(noVerdictLabel('nicht_laufbar', false), 'Cannot be rebuilt here')
})

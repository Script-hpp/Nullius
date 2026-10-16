import test from 'node:test'
import assert from 'node:assert/strict'
import { paperLabel } from './paperLabel.ts'

test('discovery ids become an arXiv label', () => {
  assert.equal(paperLabel('disc_2307_03756'), 'arXiv 2307.03756')
  assert.equal(paperLabel('disc_2410_09348_d8991277dd33d43f'), 'arXiv 2410.09348')
})

test('other ids stay readable', () => {
  assert.equal(paperLabel('kipf_gcn_2017'), 'kipf gcn 2017')
  assert.equal(paperLabel('disc_odd'), 'disc odd')
})

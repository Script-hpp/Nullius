import test from 'node:test'
import assert from 'node:assert/strict'
import { branchCount, checksFor, measuredLine, STATUS, whyNot } from './objections.ts'

const param = { type: 'parameter_change', param: 'input_window', to_value: '336', status: 'bestaetigt', measured: { mean: 0.3012, main_mean: 0.3765, delta: -0.0753, n: 3 }, attempts: [{}, {}, {}] }
const seed = { type: 'seed_variance', status: 'widerlegt', measured: { mean: 0.3761, std_pool: 0.0021, main_mean: 0.3765, n_extra: 5 } }
const rep = { type: 'reported_number', reported_value: 0.426, status: 'offen', note: 'the quote does not state the same setup as our task', measured: { mean: 0.3765, supports: 'paper' } }
const none = { type: 'not_measurable', status: 'nicht pruefbar', note: "setting 'data split' is not changeable in this task (the task fixes data and splits)" }

test('measured line per type', () => {
  assert.equal(measuredLine(param), 'input window = 336: mean 0.3012 against 0.3765 for the main code (difference -0.0753, 3 paired seeds)')
  assert.equal(measuredLine(seed), '5 extra seeds: mean 0.3761, spread over all seeds 0.0021 (main run 0.3765)')
  assert.equal(measuredLine(rep), "Reported 0.426; our measurement 0.3765 supports the paper's value")
  assert.equal(measuredLine({ type: 'parameter_change' }), 'no valid measurement')
})

test('reason shown when not measurable or open', () => {
  assert.match(whyNot(none)!, /not changeable in this task/)
  assert.match(whyNot(rep)!, /does not state the same setup/)
  assert.equal(whyNot(param), null)
  assert.match(whyNot({ type: 'not_measurable', type_reason: 'fairness critique' })!, /fairness/)
})

test('status labels and helpers', () => {
  assert.equal(STATUS.bestaetigt.label, 'Confirmed')
  assert.equal(STATUS.widerlegt.label, 'Refuted')
  assert.equal(branchCount(param), 3)
  assert.equal(checksFor({ objection_checks: [{ claim_id: 'a' }, { claim_id: 'b' }] }, 'a').length, 1)
  assert.deepEqual(checksFor({}, 'a'), [])
})

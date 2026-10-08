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

import { experimentCards, discardedFindings, funnel, provenance, measureOf } from './objections.ts'

const dossier = {
  claims: [{ id: 'a', target: 0.375, tolerance: 0.01 }],
  community: { mode: 'live', queried_at: '2026-10-08T14:58:49+00:00', findings: [{}, {}], dropped: 1, candidates: [{}, {}, {}, {}],
    dropped_items: [{ url: 'u1', reason: 'quote_not_in_page', statement: 's', quote: 'q' }, { url: 'u2', reason: 'x' }] },
  objection_checks: [
    { id: 'o1', claim_id: 'a', type: 'seed_variance', status: 'widerlegt', quote: 'q1', url: 'u', measured: { mean: 0.3793, std_pool: 0.0015, main_mean: 0.3792, n_extra: 5, n_main: 5, seeds: [5, 6, 7, 8, 9] }, attempts: [{ branch_id: 'b1' }] },
    { id: 'o2', claim_id: 'a', type: 'not_measurable', status: 'nicht pruefbar', quote: 'q2', url: 'u', note: 'classification missing or invalid', statement: 'st' },
  ],
}

test('cards only for tested objections, measure carries target band', () => {
  const cards = experimentCards(dossier, 'a')
  assert.equal(cards.length, 1)
  assert.equal(cards[0].outcome, 'The result is stable across seeds')
  assert.equal(cards[0].measure.tolerance, 0.01)
  assert.equal(cards[0].measure.points.length, 2)
  assert.match(cards[0].details[0], /Seeds 5, 6, 7, 8, 9/)
  assert.deepEqual(measureOf({ type: 'seed_variance' }, undefined).points, [])
})

test('discarded: dropped with reason plus text-only, entries without statement skipped', () => {
  const d = discardedFindings(dossier, 'a')
  assert.deepEqual(d.map((x) => x.stage), ['dropped', 'text'])
  assert.match(d[0].reason, /not in the page/)
  assert.match(d[1].reason, /Classification missing/)
})

test('funnel and provenance', () => {
  assert.deepEqual(funnel(dossier, 'a'), { found: 4, kept: 2, tested: 1, dropped: 1 })
  assert.equal(provenance(dossier.community), 'Live · 8 Oct 2026, 14:58 UTC')
  assert.equal(provenance(null), null)
})

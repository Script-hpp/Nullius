import test from 'node:test'
import assert from 'node:assert/strict'
import { TIMELAPSE_MS, closingLines, countAt, durationText, eventsForClaim, partialDossier } from './timelapse.ts'
import type { Event } from './data.ts'

test('the length is fixed: after 30 s every event is shown, whatever their number', () => {
  for (const total of [3, 40, 800]) assert.equal(countAt(TIMELAPSE_MS, total), total)
  assert.equal(countAt(TIMELAPSE_MS + 5000, 10), 10)
})

test('events are spread evenly: half the time, about half the events; start shows the first one', () => {
  assert.equal(countAt(0, 100), 1)
  assert.equal(countAt(TIMELAPSE_MS / 2, 100), 51)
  assert.equal(countAt(1, 0), 0)
})

test('many events per frame when the run was long (never slower than 30 s)', () => {
  const step = (ms: number) => countAt(ms + 16, 2000) - countAt(ms, 2000)
  assert.ok(step(10_000) >= 1 && step(10_000) <= 2 && countAt(29_000, 2000) < 2000)
})

test('duration text from the first to the last event', () => {
  const ev = (t: number) => ({ ts: t, type: 'x' }) as Event
  assert.equal(durationText([ev(0), ev(41 * 60)]), '41 minutes')
  assert.equal(durationText([ev(100), ev(130)]), '30 seconds')
  assert.equal(durationText([ev(0), ev(2 * 3600 + 5 * 60)]), '2 h 5 min')
  assert.equal(durationText([ev(0)]), null)
})

test('events of other claims are dropped, claim-less events stay', () => {
  const evs = [{ ts: 0, type: 'run_start' }, { ts: 1, type: 'attempt', claim_id: 'a' }, { ts: 2, type: 'attempt', claim_id: 'b' }] as Event[]
  assert.deepEqual(eventsForClaim(evs, 'a').map((e) => [e.type, e.claim_id ?? null]), [['run_start', null], ['attempt', 'a']])
})

test('partial dossier: measurements arrive one by one, the verdict only with its event, the real dossier at the end', () => {
  const dossier = { claims: [{ id: 'a' }], attempts: [{ measured: 1 }], verdicts: [{ claim_id: 'a', urteil: 'reproduziert' }] }
  const seen = [
    { ts: 0, type: 'attempt', claim_id: 'a', seed: 0, measured: 0.8, status: 'stabil' },
    { ts: 1, type: 'attempt', claim_id: 'a', seed: 1, measured: null, status: 'fehler' },
  ] as Event[]
  const mid = partialDossier(dossier, 'a', seen, false)
  assert.equal(mid.attempts.length, 2)
  assert.deepEqual(mid.verdicts, [])
  const withVerdict = partialDossier(dossier, 'a', [...seen, { ts: 2, type: 'verdict', claim_id: 'a' }] as Event[], false)
  assert.equal(withVerdict.verdicts.length, 1)
  assert.equal(partialDossier(dossier, 'a', seen, true), dossier)
})

test('closing lines carry the certificate sentence verbatim and the fidelity note', () => {
  const sentence = 'Reproduced with our setup (target value reached): measured 82.1% against the reported 81.5%.'
  const c = closingLines({ label: 'REPRODUCED', sentence, fidelityUnchecked: true, overall: 'Paper reproduced' })
  assert.deepEqual(c, { label: 'REPRODUCED', sentence, note: 'Method fidelity not checked.' })
  assert.equal(closingLines({ label: 'REPRODUCED', sentence, fidelityUnchecked: false, overall: 'Target reached with different model' }).note, 'Target reached with different model')
  assert.equal(closingLines({ label: 'NOT REPRODUCED', sentence, fidelityUnchecked: false }).note, null)
})

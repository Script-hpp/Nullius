import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { measuredAttempts } from './measurements.ts'
import { measurements, dossierAt } from './film.ts'

const root = new URL('../public/data/', import.meta.url)
const glinear = JSON.parse(readFileSync(new URL('disc_2501_01087/20261010-161916/dossier.json', root), 'utf8'))
const events = readFileSync(new URL('disc_2501_01087/20261010-161916/events.jsonl', root), 'utf8').trim().split('\n').map((line) => JSON.parse(line))
const claimId = glinear.claims[0].id

test('GLinear verdict uses the final three seeds, excluding other candidates and pre-repair code', () => {
  const selected = measuredAttempts(glinear, claimId)
  assert.deepEqual(selected.map((a) => a.seed).sort(), [0, 1, 2])
  assert.deepEqual([...new Set(selected.map((a) => a.code_hash))], ['67dae24762de'])
  const mean = selected.reduce((sum, a) => sum + a.measured, 0) / selected.length
  assert.ok(Math.abs(mean - glinear.verdicts[0].measured_mean) < 1e-12)
  assert.equal(glinear.attempts.filter((a) => a.measured != null).length, 6)
  assert.deepEqual(measurements(events, claimId, glinear), selected.map((a) => a.measured))
  assert.equal(dossierAt(glinear, claimId, events, 6, true).attempts.length, 3)
  assert.equal(dossierAt(glinear, claimId, events, 1, false).attempts.length, 1)
})

test('selected measurements agree with each published verdict, including legacy runs', () => {
  for (const paper of readdirSync(root, { withFileTypes: true }).filter((x) => x.isDirectory())) {
    const dir = new URL(`${paper.name}/`, root)
    for (const run of readdirSync(dir, { withFileTypes: true }).filter((x) => x.isDirectory())) {
      const path = new URL(`${run.name}/dossier.json`, dir)
      let dossier
      try { dossier = JSON.parse(readFileSync(path, 'utf8')) } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue
        throw error
      }
      for (const verdict of dossier.verdicts ?? []) {
        const values = measuredAttempts(dossier, verdict.claim_id).map((a) => a.measured)
        if (!values.length) continue
        assert.ok(Math.abs(values.reduce((a, b) => a + b, 0) / values.length - verdict.measured_mean) < 1e-9, path.pathname)
      }
    }
  }
})

test('no selected measurement does not silently promote rejected attempts; claims and invalid values stay excluded', () => {
  const dossier = { attempts: [
    { claim_id: 'c', chosen: false, measured: 0.3 },
    { claim_id: 'c', chosen: true, measured: null },
    { claim_id: 'other', chosen: true, measured: 0.4 },
  ] }
  assert.deepEqual(measuredAttempts(dossier, 'c'), [])
  assert.deepEqual(measuredAttempts({ attempts: [{ claim_id: 'c', measured: 0 }, { claim_id: 'c', measured: NaN }] }, 'c').map((a) => a.measured), [0])
})

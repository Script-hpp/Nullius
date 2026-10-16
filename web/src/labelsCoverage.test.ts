import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { ATTEMPT, METHOD, OVERALL, RESULT } from './labels.ts'

// Every internal (German) verdict, status or class that the exported data can hand to a page must have an English label. de2en would otherwise
// fall back to the raw key in capitals.
const DATA = 'public/data'
const json = (p: string) => JSON.parse(readFileSync(p, 'utf8'))
const english = (label: string) => /^[A-Z][A-Z ]+$/.test(label)

test('every verdict class in benchmark.json maps to an English label', () => {
  const b = json(join(DATA, 'benchmark.json'))
  const keys = new Set<string>()
  for (const v of b.versions) for (const u of v.units) {
    keys.add(u.reference_verdict)
    for (const arm of Object.values<any>(u.arms)) { keys.add(arm.majority); arm.verdicts.forEach((x: string) => keys.add(x)) }
    for (const arr of Object.values<any>(u.placebo)) arr.forEach((x: string) => keys.add(x))
  }
  assert.ok(keys.size > 5)
  for (const k of keys) assert.ok(RESULT[k], `no label for benchmark value ${JSON.stringify(k)}`)
  assert.equal(RESULT['keine Mehrheit'].label, 'NO MAJORITY')
  assert.equal(RESULT['Ausfall'].label, 'FAILED RUN')
})

test('every verdict, overall statement, method check and attempt status in the catalog and run dossiers has a label', () => {
  const cat = json(join(DATA, 'catalog.json'))
  for (const e of (Array.isArray(cat) ? cat : cat.entries)) {
    if (!e.verdict) continue
    assert.ok(RESULT[e.verdict.urteil], `catalog urteil ${e.verdict.urteil}`)
    if (e.verdict.gesamtaussage) assert.ok(OVERALL[e.verdict.gesamtaussage], `catalog gesamtaussage ${e.verdict.gesamtaussage}`)
  }
  for (const paper of readdirSync(DATA)) for (const ts of (() => { try { return readdirSync(join(DATA, paper)) } catch { return [] } })()) {
    let d: any
    try { d = json(join(DATA, paper, ts, 'dossier.json')) } catch { continue }
    for (const v of d.verdicts ?? []) {
      assert.ok(RESULT[v.urteil], `${paper}/${ts} urteil ${v.urteil}`)
      if (v.methode) assert.ok(METHOD[v.methode], `${paper}/${ts} methode ${v.methode}`)
    }
    for (const a of d.attempts ?? []) assert.ok(ATTEMPT[a.status], `${paper}/${ts} status ${a.status}`)
  }
})

test('all labels read as English capitals, no German leftovers', () => {
  for (const map of [RESULT, ATTEMPT, METHOD]) for (const v of Object.values(map)) assert.ok(english(v.label), v.label)
})

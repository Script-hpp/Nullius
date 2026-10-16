import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { claimLine, juryEntries, splitFirstSentence } from './landingBottom.ts'
import type { CatalogEntry } from './catalog'
import { LIMITS, SCOPE_LIMIT } from './data/benchmark.ts'

const entries: CatalogEntry[] = JSON.parse(readFileSync(new URL('../public/data/catalog.json', import.meta.url), 'utf8'))

test('only jury-ready entries are picked; null catalog gives none', () => {
  const picked = juryEntries(entries)
  assert.ok(picked.length > 0)
  assert.ok(picked.every((e) => e.jury?.ready))
  assert.deepEqual(juryEntries(null), [])
  assert.deepEqual(juryEntries([]), [])
})

test('claim line comes from the exported claim', () => {
  const byId = (id: string) => entries.find((e) => e.id === id)!
  assert.equal(claimLine(byId('disc_2307_03756')), 'FITS reaches 0.372 MSE on ETTh1')
  assert.equal(claimLine(byId('disc_1609_02907')), 'GCN reaches 81.5% accuracy on Cora')
})

test('first sentence split keeps all text', () => {
  for (const l of [...LIMITS, SCOPE_LIMIT]) {
    const { lead, rest } = splitFirstSentence(l.text)
    assert.ok(lead.length > 0)
    assert.equal(rest ? `${lead} ${rest}` : lead, l.text)
  }
  assert.deepEqual(splitFirstSentence('One only.'), { lead: 'One only.', rest: '' })
})

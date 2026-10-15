import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { catalogFilterFromHash, matchesCatalogFilter, recordedDate } from './catalogFilter.ts'
import type { CatalogEntry } from './catalog'

const entries: CatalogEntry[] = JSON.parse(readFileSync(new URL('../public/data/catalog.json', import.meta.url), 'utf8'))

test('all outcomes link reaches setup failures and unchecked candidates beyond the featured selection', () => {
  assert.equal(catalogFilterFromHash('#catalog?view=all'), 'all')
  assert.equal(catalogFilterFromHash('#catalog'), 'featured')
  assert.equal(catalogFilterFromHash('#catalog?view=unknown'), 'featured')
  const all = entries.filter((e) => matchesCatalogFilter(e, 'all'))
  assert.equal(all.length, entries.length)
  assert.ok(all.length > entries.filter((e) => matchesCatalogFilter(e, 'featured')).length)
  assert.ok(entries.filter((e) => matchesCatalogFilter(e, 'failed')).some((e) => e.verdict?.urteil === 'nicht reproduzierbar (Setup)'))
  assert.ok(entries.filter((e) => matchesCatalogFilter(e, 'unchecked')).every((e) => e.status === 'ungeprueft'))
  assert.ok(entries.filter((e) => matchesCatalogFilter(e, 'unchecked')).length > 0)
  assert.equal(matchesCatalogFilter({ status: 'lauf_fehlgeschlagen', verdict: null } as CatalogEntry, 'failed'), true)
})

test('recorded results are listed with their date and note, and only jury-ready entries stay startable', () => {
  const recorded = entries.filter((e) => e.recorded)
  assert.deepEqual(recorded.map((e) => e.id).sort(), ['disc_2303_06053', 'disc_2405_00946', 'disc_2410_09348_d8991277dd33d43f'])
  for (const e of recorded) {
    assert.ok(e.run_id && e.recorded!.date && e.recorded!.note.length > 40)
    assert.equal(e.jury?.ready, false)
    assert.equal(matchesCatalogFilter(e, 'featured'), true)
  }
  assert.equal(recordedDate('2026-10-09'), '9 Oct 2026')
  assert.deepEqual(entries.filter((e) => e.jury?.ready).map((e) => e.id).sort(), ['disc_1609_02907', 'disc_2307_03756', 'disc_2501_01087'])
})

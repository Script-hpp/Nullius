import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { catalogFilterFromHash, matchesCatalogFilter } from './catalogFilter.ts'
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

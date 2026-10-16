import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { cleanTitle } from './title.ts'

test('inline math in a title becomes plain text', () => {
  assert.equal(cleanTitle('FITS: Modeling Time Series with $10k$ Parameters'), 'FITS: Modeling Time Series with 10k Parameters')
  assert.equal(cleanTitle('Scaling to $O(n^{2})$ nodes'), 'Scaling to O(n^2) nodes')
  assert.equal(cleanTitle('A $\\mathrm{GCN}$ for $\\alpha$-graphs'), 'A GCN for α-graphs')
  assert.equal(cleanTitle('Losses of 5\\% on Cora'), 'Losses of 5% on Cora')
})

test('titles without math and prose dollar signs stay as they are', () => {
  assert.equal(cleanTitle('Semi-Supervised Classification with Graph Convolutional Networks'), 'Semi-Supervised Classification with Graph Convolutional Networks')
  assert.equal(cleanTitle('Fixing 162 out of 337 bugs for $0.42 each using ChatGPT'), 'Fixing 162 out of 337 bugs for $0.42 each using ChatGPT')
})

// Every title the site can show: index, catalog and each dossier summary. None may keep a LaTeX mark after cleaning.
test('no exported title keeps LaTeX marks', () => {
  const dir = join(import.meta.dirname, '..', 'public', 'data')
  const titles: string[] = []
  for (const f of ['index.json', 'catalog.json']) {
    const p = join(dir, f)
    if (existsSync(p)) for (const x of JSON.parse(readFileSync(p, 'utf8'))) titles.push(x.title)
  }
  for (const d of readdirSync(dir, { withFileTypes: true })) {
    if (!d.isDirectory()) continue
    for (const t of readdirSync(join(dir, d.name))) {
      const p = join(dir, d.name, t, 'dossier.json')
      if (!existsSync(p)) continue
      const dossier = JSON.parse(readFileSync(p, 'utf8'))
      for (const k of ['title', 'paper_title']) if (typeof dossier[k] === 'string') titles.push(dossier[k])
      if (typeof dossier.summary?.title === 'string') titles.push(dossier.summary.title)
    }
  }
  assert.ok(titles.length > 10)
  for (const t of titles) assert.ok(!/\$[^$\s][^$]*\$|\\[A-Za-z]+\{/.test(cleanTitle(t)), t)
})

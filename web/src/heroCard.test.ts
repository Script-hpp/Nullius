import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { heroCard, rampValue, recordedDate } from './heroCard.ts'

const FEATURED = 'disc_2307_03756--20261014-080542'
const dir = new URL('../public/data/disc_2307_03756/20261014-080542/', import.meta.url)
const dossier = JSON.parse(readFileSync(new URL('dossier.json', dir), 'utf8'))
const catalog = JSON.parse(readFileSync(new URL('../public/data/catalog.json', import.meta.url), 'utf8'))
const entry = catalog.find((e: { id: string }) => e.id === 'disc_2307_03756')
const run = { ref: { title: 'FITS: Modeling Time Series with $10k$ Parameters', timestamp: FEATURED.split('--')[1] }, dossier }

test('card of the featured run carries the exported numbers', () => {
  const c = heroCard(run, entry, 3)!
  assert.equal(c.name, 'FITS')
  assert.equal(c.line, 'FITS: forecast error 0.372 on ETTh1')
  assert.equal(c.claimed, '0.3720')
  assert.equal(c.measured, '0.3720')
  assert.equal(c.urteil, 'reproduziert')
  assert.equal(c.meta, '3 seeds · recorded 14 Oct 2026 · live Tavily calls · $0.31')
})

test('no claim or verdict gives no card', () => {
  assert.equal(heroCard(null, entry, 3), null)
  assert.equal(heroCard({ ref: {}, dossier: { claims: [], verdicts: [] } }, entry, 3), null)
})

test('meta leaves out what is missing', () => {
  const c = heroCard({ ref: { title: 'X' }, dossier: { claims: dossier.claims, verdicts: dossier.verdicts, sources: [] } }, undefined, 2)!
  assert.equal(c.meta, '2 seeds')
  assert.equal(c.name, 'X')
})

test('recorded date parses run timestamps only', () => {
  assert.equal(recordedDate('20261014-080542'), '14 Oct 2026')
  assert.equal(recordedDate('2026'), null)
  assert.equal(recordedDate(undefined), null)
  assert.equal(recordedDate('20261314-000000'), null)
})

test('ramp ends exactly on the target and keeps decimals', () => {
  assert.equal(rampValue(0.372031, 0, 700), 0)
  assert.equal(rampValue(0.372031, 700, 700), 0.372031)
  const mid = rampValue(0.372031, 350, 700)
  assert.ok(mid > 0.2 && mid < 0.372031)
  assert.ok(readdirSync(dir).length > 0)
})

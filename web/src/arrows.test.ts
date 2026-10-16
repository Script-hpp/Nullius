import test from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { arrowParts } from './components/arrowParts.ts'

test('arrowParts splits on the unicode arrows and trims the spaces around them', () => {
  assert.deepEqual(arrowParts('input → hidden = 16 → output'), ['input', 'hidden = 16', 'output'])
  assert.deepEqual(arrowParts('no arrow here'), ['no arrow here'])
  assert.deepEqual(arrowParts('A ⟶ B'), ['A', 'B'])
})

// No user-facing source shows a unicode arrow: pages use the ArrowRight icon. Exceptions: the film components (not part of the pages), the
// film/latex helpers that parse arrows out of text, and tests.
const walk = (dir: string): string[] => readdirSync(dir).flatMap((f) => {
  const p = join(dir, f)
  return statSync(p).isDirectory() ? walk(p) : [p]
})
test('no page component or page string carries a unicode arrow or &rarr;', () => {
  const skip = (p: string) => p.includes('/components/film/') || /\.test\.ts$/.test(p) || /\/(film|latex|arrowParts)\.ts$/.test(p)
  const bad = walk('src').filter((p) => /\.(tsx?|css)$/.test(p) && !skip(p)).filter((p) => /→|⟶|&rarr;|\\u2192/.test(readFileSync(p, 'utf8')))
  assert.deepEqual(bad, [])
})

test('exported summary data shown on pages carries no unicode arrow', () => {
  for (const f of ['catalog.json', 'benchmark.json', 'index.json']) {
    let txt = ''
    try { txt = readFileSync(join('public/data', f), 'utf8') } catch { continue }
    assert.ok(!/→|⟶/.test(txt), f)
  }
})

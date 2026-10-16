import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync } from 'node:fs'
import { FINDINGS, INTRO, partA, partB, visibleText } from './findings.ts'

const data = new URL('../public/data/', import.meta.url)
const repo = new URL('../../', import.meta.url)
const words = (s: string) => s.split(/\s+/).filter(Boolean).length

test('part A has five paper findings, part B four checker findings; ids unique', () => {
  assert.equal(partA.length, 5)
  assert.equal(partB.length, 4)
  assert.equal(new Set(FINDINGS.map((f) => f.id)).size, FINDINGS.length)
})

test('every finding: short headline, two short sentences at most, details and a valid link', () => {
  for (const f of FINDINGS) {
    assert.ok(f.tag && f.headline && f.text && f.details.length >= 2 && f.links.length >= 1, f.id)
    assert.ok(words(f.headline) <= 11, `${f.id}: headline too long`)
    assert.ok(words(f.text) <= 40, `${f.id}: text too long`)
    assert.ok((f.text.match(/[.!?](\s|$)/g) ?? []).length <= 3, `${f.id}: too many sentences`)
    for (const d of f.details) assert.ok(d.label && d.text.length > 20, `${f.id}: ${d.label}`)
    for (const c of f.commits) assert.match(c, /^[0-9a-f]{7}$/, f.id)
  }
  for (const f of partA) assert.ok(f.left && f.right, `${f.id}: part A needs the side-by-side`)
})

test('run links point at an exported run; hash routes are known; file links point at a file in the repo', () => {
  for (const f of FINDINGS) {
    for (const l of f.links) {
      const run = /^#\/run\/([^?]+)--(\d{8}-\d{6})$/.exec(l.href)
      if (run) {
        assert.ok(readdirSync(new URL(`${run[1]}/`, data)).includes(run[2]), `${f.id}: run not exported`)
      } else if (l.href.startsWith('#')) {
        assert.ok(['#benchmark', '#catalog', '#discover', '#findings'].includes(l.href), `${f.id}: unknown route ${l.href}`)
      } else {
        assert.ok(l.repoPath && l.href.endsWith(l.repoPath), `${f.id}: external link needs repoPath`)
        assert.ok(existsSync(new URL(l.repoPath, repo)), `${f.id}: ${l.repoPath} missing`)
      }
    }
  }
})

test('visible text with details closed stays short (about two minutes)', () => {
  assert.ok(words(visibleText()) <= 520, `${words(visibleText())} words`)
  assert.ok(INTRO.summary.length > 20)
})

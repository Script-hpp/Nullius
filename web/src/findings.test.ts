import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readdirSync } from 'node:fs'
import { FINDINGS } from './findings.ts'

const data = new URL('../public/data/', import.meta.url)
const repo = new URL('../../', import.meta.url)

test('four findings, each with title, text, number pair and a valid link', () => {
  assert.equal(FINDINGS.length, 4)
  assert.equal(new Set(FINDINGS.map((f) => f.id)).size, 4)
  for (const f of FINDINGS) {
    assert.ok(f.paper && f.dataset && f.title.length > 10 && f.text.length > 40, f.id)
    assert.ok(f.says.value && f.found.value && f.link.label, f.id)
  }
})

test('run links point at an exported run; file links point at a file in the repo', () => {
  for (const f of FINDINGS) {
    const run = /^#\/run\/([^?]+)--(\d{8}-\d{6})$/.exec(f.link.href)
    if (run) {
      assert.ok(readdirSync(new URL(`${run[1]}/`, data)).includes(run[2]), `${f.id}: run not exported`)
    } else {
      assert.ok(f.repoPath, `${f.id}: external link needs repoPath`)
      assert.ok(f.link.href.endsWith(f.repoPath!), f.id)
      assert.ok(existsSync(new URL(f.repoPath!, repo)), `${f.id}: ${f.repoPath} missing`)
    }
  }
})

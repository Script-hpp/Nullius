import test from 'node:test'
import assert from 'node:assert/strict'
import { parseTab, tabHash } from './runtabs.ts'

test('new tab names pass through', () => {
  for (const t of ['overview', 'how', 'sources', 'code', 'limits']) assert.equal(parseTab(t).tab, t)
})

test('old tab names map to the nearest new tab', () => {
  assert.deepEqual(parseTab('log'), { tab: 'how', how: 'log' })
  assert.deepEqual(parseTab('graph'), { tab: 'how', how: 'graph' })
  assert.equal(parseTab('evidence').tab, 'sources')
  assert.equal(parseTab('tree').tab, 'code')
  assert.equal(parseTab('files').tab, 'code')
})

test('unknown or missing tab opens the overview', () => {
  assert.equal(parseTab(null).tab, 'overview')
  assert.equal(parseTab('nonsense').tab, 'overview')
})

test('tabHash keeps the run and drops other query parts', () => {
  assert.equal(tabHash('#/run/abc--1?play=1', 'code'), '#/run/abc--1?tab=code')
  assert.equal(tabHash('#/run/abc--1?tab=code', 'overview'), '#/run/abc--1')
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { MODE_AUTHOR, MODE_PAPER, runMode } from './mode.ts'

const ev = (type: string, o: object = {}) => ({ ts: 0, type, ...o })

test('confirmed authors-code mode from the live mode event', () => {
  const m = runMode(null, [ev('run_start', { mode_requested: 'nachbau_autorencode' }), ev('mode', { mode: 'nachbau_autorencode', fallback_reason: null })])
  assert.equal(m?.label, MODE_AUTHOR)
  assert.equal(m?.planned, false)
  assert.match(m?.explain ?? '', /never run/)
  assert.match(m?.explain ?? '', /harness/)
})

test('requested but not yet confirmed is shown as planned', () => {
  const m = runMode(null, [ev('run_start', { mode_requested: 'nachbau_autorencode' })])
  assert.equal(m?.key, 'nachbau_autorencode')
  assert.equal(m?.planned, true)
})

test('fallback is labelled as paper-only, never as authors code', () => {
  const m = runMode(null, [ev('run_start', { mode_requested: 'nachbau_autorencode' }), ev('mode', { mode: 'nachbau', fallback_reason: 'no repository found or clone failed' })])
  assert.equal(m?.label, MODE_PAPER)
  assert.equal(m?.key, 'nachbau')
  assert.match(m?.explain ?? '', /No authors' code was used because no usable repository was found/)
})

test('dossier alone: mode field and fallback warning', () => {
  assert.equal(runMode({ mode: 'nachbau_autorencode' }, [])?.label, MODE_AUTHOR)
  const w = 'Autorencode nicht verfügbar (no repository found or clone failed): Lauf als reiner Nachbau aus dem Paper.'
  const m = runMode({ mode: 'nachbau', warnings: [w] }, [])
  assert.equal(m?.label, MODE_PAPER)
  assert.match(m?.explain ?? '', /No authors' code was used/)
})

test('older runs without mode field are paper-only; nachlauf shows nothing', () => {
  assert.equal(runMode({ paper_id: 'kipf_gcn_2017' }, [])?.label, MODE_PAPER)
  assert.equal(runMode({ mode: 'nachlauf' }, []), null)
  assert.equal(runMode(null, []), null)
})

test('queued job event shows the requested mode as planned before run_start', () => {
  assert.equal(runMode(null, [ev('queued', { mode_requested: 'nachbau_autorencode' })])?.planned, true)
  assert.equal(runMode(null, [ev('queued', { mode_requested: 'nachbau' })])?.label, MODE_PAPER)
})

test('fallback warning is shown in English', async () => {
  const { translateWarning } = await import('./labels.ts')
  const t = translateWarning('Autorencode nicht verfügbar (no repository found or clone failed): Lauf als reiner Nachbau aus dem Paper.')
  assert.match(t, /no usable repository was found/)
  assert.match(t, /rebuild from the paper alone/)
})

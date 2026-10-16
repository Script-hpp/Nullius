import test from 'node:test'
import assert from 'node:assert/strict'
import { easeToward, sceneProgress, sceneTime, scrollLengthVh } from './progress.ts'
import { FRAME_BG, SCENES, SCENE_IDS, paletteFor } from './scenes.ts'
import { PAL } from './palette.ts'

test('scroll length follows the duration, bounded', () => {
  assert.equal(scrollLengthVh(1000), 1.2)
  assert.equal(scrollLengthVh(15000), 2.5)
  assert.equal(scrollLengthVh(60000), 3.5)
})

test('progress is clamped and zero without scroll travel', () => {
  assert.equal(sceneProgress(300, 2000, 800), 0)
  assert.equal(sceneProgress(0, 2000, 800), 0)
  assert.equal(sceneProgress(-600, 2000, 800), 0.5)
  assert.equal(sceneProgress(-5000, 2000, 800), 1)
  assert.equal(sceneProgress(-10, 800, 800), 0)
})

test('film time scrubs with progress and holds both ends', () => {
  const o = { still: false }
  assert.equal(sceneTime(0, 20000, o), 0)
  assert.equal(sceneTime(0.04, 20000, o), 0)
  assert.equal(sceneTime(0.5, 20000, o), 10000)
  assert.ok(Math.abs(sceneTime(0.96, 20000, o) - 20000) < 1e-6)
  assert.equal(sceneTime(1, 20000, o), 20000)
  assert.equal(sceneTime(-1, 20000, o), 0)
  assert.ok(sceneTime(0.3, 20000, o) < sceneTime(0.4, 20000, o))
})

test('reduced motion and phones show the poster frame whatever the progress', () => {
  for (const p of [0, 0.5, 1]) assert.equal(sceneTime(p, 21000, { still: true, poster: 19500 }), 19500)
  assert.equal(sceneTime(0.5, 9000, { still: true }), 8800)
  assert.equal(sceneTime(0.5, 9000, { still: true, poster: 99999 }), 9000)
})

test('easing moves towards the target and snaps', () => {
  const a = easeToward(0, 1000, 16)
  assert.ok(a > 0 && a < 1000)
  assert.equal(easeToward(999.5, 1000, 16), 1000)
  assert.equal(easeToward(500, 100, 0), 500)
  assert.ok(easeToward(500, 100, 16) < 500)
})

test('every scene has a poster inside its duration and a plain description', () => {
  for (const id of SCENE_IDS) {
    const s = SCENES[id]
    assert.ok(s.poster > 0 && s.poster <= s.duration, id)
    assert.ok(s.label.length > 20 && s.caption.length > 20, id)
  }
})

test('dark mode forces the night palette, light mode keeps the shot palette', () => {
  assert.equal(paletteFor('paper', true), 'night')
  assert.equal(paletteFor('paper', false), 'paper')
  assert.equal(paletteFor('night', false), 'night')
  assert.equal(FRAME_BG.paper, PAL.paper.bg)
  assert.equal(FRAME_BG.night, PAL.night.bg)
})

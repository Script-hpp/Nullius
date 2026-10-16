import test from 'node:test'
import assert from 'node:assert/strict'
import { VISIBLE_AT, advance, isFinished, isVisibleEnough, playTime, stillTime } from './progress.ts'
import { FRAME_BG, SCENES, SCENE_IDS, paletteFor, stillSrc } from './scenes.ts'
import { PAL } from './palette.ts'

test('play time starts at start, runs in real time and holds the last frame', () => {
  assert.equal(playTime(0, 2500, 21000), 2500)
  assert.equal(playTime(1000, 2500, 21000), 3500)
  assert.equal(playTime(-5, 2500, 21000), 2500)
  assert.equal(playTime(99999, 2500, 21000), 21000)
})

test('finished only at the end', () => {
  assert.equal(isFinished(20999, 21000), false)
  assert.equal(isFinished(21000, 21000), true)
})

test('a tick advances while playing, stands still while paused, clamps at the end', () => {
  assert.equal(advance(1000, 16, 9000, true), 1016)
  assert.equal(advance(1000, 16, 9000, false), 1000)
  assert.equal(advance(8995, 16, 9000, true), 9000)
  assert.equal(advance(1000, 5000, 9000, true), 1100)   // a long gap is capped
  assert.equal(advance(1000, -3, 9000, true), 1000)
})

test('pause and resume continue from where they stopped', () => {
  let t = advance(0, 16, 9000, true)
  t = advance(t, 16, 9000, false)
  t = advance(t, 16, 9000, true)
  assert.equal(t, 32)
})

test('visibility threshold is 70% of the frame', () => {
  assert.equal(VISIBLE_AT, 0.7)
  assert.equal(isVisibleEnough(0.69), false)
  assert.equal(isVisibleEnough(0.7), true)
  assert.equal(isVisibleEnough(1), true)
})

test('the still (reduced motion) shows the poster frame', () => {
  assert.equal(stillTime(21000, 19500), 19500)
  assert.equal(stillTime(9000), 8800)
  assert.equal(stillTime(9000, 99999), 9000)
})

test('every scene starts inside its duration, has a poster after it and a plain description', () => {
  for (const id of SCENE_IDS) {
    const s = SCENES[id]
    assert.ok(s.start >= 0 && s.start < s.duration, id)
    assert.ok(s.poster > s.start && s.poster <= s.duration, id)
    assert.ok(s.label.length > 20 && s.caption.length > 20, id)
    assert.ok(!/scroll/i.test(s.label + s.caption), id)
  }
})

test('stills are named by scene, kind and palette', () => {
  assert.equal(stillSrc('opening', 'start', 'paper'), 'img/scenes/opening-start-paper.webp')
  assert.equal(stillSrc('placebo', 'poster', 'night'), 'img/scenes/placebo-poster-night.webp')
})

test('dark mode forces the night palette, light mode keeps the shot palette', () => {
  assert.equal(paletteFor('paper', true), 'night')
  assert.equal(paletteFor('paper', false), 'paper')
  assert.equal(paletteFor('night', false), 'night')
  assert.equal(FRAME_BG.paper, PAL.paper.bg)
  assert.equal(FRAME_BG.night, PAL.night.bg)
})

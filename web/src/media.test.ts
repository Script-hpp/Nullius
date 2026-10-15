import test from 'node:test'
import assert from 'node:assert/strict'
import { durationLabel, FILM_SECONDS, loopShouldPlay, nextFocusIndex } from './media.ts'

test('film duration label is m:ss', () => {
  assert.equal(durationLabel(FILM_SECONDS), '2:38')
  assert.equal(durationLabel(65), '1:05')
})

test('loop plays only when visible and motion is allowed (reduced motion -> poster only)', () => {
  assert.equal(loopShouldPlay(true, false), true)
  assert.equal(loopShouldPlay(false, false), false)
  assert.equal(loopShouldPlay(true, true), false)
})

test('focus trap wraps in both directions', () => {
  assert.equal(nextFocusIndex(3, 2, false), 0)
  assert.equal(nextFocusIndex(3, 0, true), 2)
  assert.equal(nextFocusIndex(3, -1, false), 0)
  assert.equal(nextFocusIndex(0, 0, false), null)
})

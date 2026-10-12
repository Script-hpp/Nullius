import test from 'node:test'
import assert from 'node:assert/strict'
import { collectionHash, rememberRunOrigin, runOrigin, takeCollectionPosition } from './runNavigation.ts'

test('return from a run keeps its collection filter and scroll position, including after tab changes', () => {
  rememberRunOrigin('#/run/check-one?play=1', '#catalog?view=all', 810)
  assert.deepEqual(runOrigin('check-one'), { hash: '#catalog?view=all', scrollTop: 810 })
  assert.equal(takeCollectionPosition('#catalog?view=all'), 810)
  assert.equal(takeCollectionPosition('#catalog?view=all'), null)
  rememberRunOrigin('#/run/check-one?tab=log', '#overview', 320)
  assert.deepEqual(runOrigin('check-one'), { hash: '#overview', scrollTop: 320 })
})

test('direct links use a sensible default and external or unknown origins are ignored', () => {
  for (const hash of ['https://evil.example', '#catalog?view=invalid', '#/run/other', '#discover']) {
    assert.equal(collectionHash(hash), null)
    rememberRunOrigin('#/run/direct', hash, 250)
  }
  rememberRunOrigin('#/run/direct', '#catalog', NaN)
  assert.deepEqual(runOrigin('direct'), { hash: '#overview', scrollTop: 0 })
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { badgeMarkdown, badgeUrls } from './badge.ts'

test('markdown snippet links the badge image to the certificate', () => {
  assert.equal(
    badgeMarkdown('https://example.org', 'kipf_gcn_2017', '20260929-130133'),
    '[![Nullius](https://example.org/data/kipf_gcn_2017/20260929-130133/badge.svg)](https://example.org/#/run/kipf_gcn_2017--20260929-130133)',
  )
})

test('trailing slashes in the base URL are dropped', () => {
  assert.equal(badgeUrls('https://example.org//', 'p', 't').image, 'https://example.org/data/p/t/badge.svg')
})

import { badgeNote } from './badge.ts'

test('reproduced note names the target value and that method fidelity is not checked', () => {
  const n = badgeNote('reproduziert')
  assert.match(n.text, /Reproduced with our setup \(target value reached; method fidelity not checked\)/)
  assert.equal(n.caution, false)
})

test('every non-reproduced verdict says it is not a finding about the paper, except partial', () => {
  for (const u of ['nicht reproduziert', 'nicht reproduzierbar (Setup)', 'Agenten-Fehler vermutet']) {
    const n = badgeNote(u)
    assert.match(n.text, /not a finding about the paper/)
    assert.equal(n.caution, true)
  }
  assert.match(badgeNote('teilweise').text, /^Partially reproduced with our setup/)
})

test('catalog run whose code did not implement the method is not assessable', () => {
  const n = badgeNote('reproduziert', 'nicht bewertbar (Methode nicht umgesetzt)', true)
  assert.match(n.text, /not assessable/)
  assert.doesNotMatch(n.text, /^Reproduced/)
})

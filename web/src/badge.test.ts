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

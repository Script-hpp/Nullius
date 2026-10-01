import test from 'node:test'
import assert from 'node:assert/strict'
import { authorLine, findHead, firstSentence, monogram, whenLine } from './paperhead.ts'

test('monogram: initials without linking words, at most four letters', () => {
  assert.equal(monogram('University of Amsterdam'), 'UA')
  assert.equal(monogram('Technical University of Munich'), 'TUM')
  assert.equal(monogram('The Chinese University of Hong Kong'), 'CUHK')
  assert.equal(monogram('Mohamed bin Zayed University of Artificial Intelligence'), 'MBZU')
  assert.equal(monogram('Canadian Institute for Advanced Research (CIFAR)'), 'CIAR')
  assert.equal(monogram('Mila – Québec Artificial Intelligence Institute'), 'MQAI')
  assert.equal(monogram('Microsoft Research'), 'MR')
})

test('monogram: one word gives three letters, nothing gives a placeholder', () => {
  assert.equal(monogram('Miniml.AI'), 'Min')
  assert.equal(monogram('google'), 'Goo')
  assert.equal(monogram('of the'), '?')
})

test('authorLine counts the names it hides', () => {
  assert.deepEqual(authorLine(['A', 'B'], 5), { shown: ['A', 'B'], more: 0 })
  assert.deepEqual(authorLine(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 5), { shown: ['A', 'B', 'C', 'D', 'E'], more: 2 })
})

test('whenLine prefers the venue, falls back to the year, else nothing', () => {
  assert.equal(whenLine({ year: 2016, venue: { name: 'ICLR 2017', quote: 'x' } }), 'ICLR 2017')
  assert.equal(whenLine({ year: 2016 }), '2016')
  assert.equal(whenLine({ year: null }), null)
})

test('findHead finds a paper by its arXiv id or by any run/catalog id of it', () => {
  const all = { '1811.05868': { summary: 's', source: 'x', paper_ids: ['shchur_pitfalls_2018', 'shchur_gat_2018'] }, '2511.08094': { summary: 't', source: 'x', paper_ids: ['disc_2511_08094'] } }
  assert.equal(findHead(all, 'shchur_gat_2018')?.arxiv_id, '1811.05868')
  assert.equal(findHead(all, '2511.08094')?.summary, 't')
  assert.equal(findHead(all, 'unknown'), undefined)
  assert.equal(findHead(all, null), undefined)
})

test('firstSentence cuts at the first full stop', () => {
  assert.equal(firstSentence('The paper proposes X. The data is Y.'), 'The paper proposes X.')
  assert.equal(firstSentence('No full stop'), 'No full stop')
})

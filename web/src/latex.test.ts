import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { splitFormulas, toLatex } from './latex.ts'
import { planItems } from './film.ts'

const KIPF = readFileSync(new URL('../public/data/kipf_gcn_2017/20260929-130133/plan.md', import.meta.url), 'utf8')

test('accents become \\tilde and \\hat, exponents get braces, -1/2 stays as written', () => {
  assert.equal(toLatex('Ã = A + I'), '\\tilde{A} = A + I')
  assert.equal(toLatex('Â = D̃^(-1/2) Ã D̃^(-1/2)'), '\\hat{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}')
  assert.equal(toLatex('x^2 + y^-3'), 'x^{2} + y^{-3}')
})

test('arrows become \\to inside a formula, TeX special characters are escaped, everything else is unchanged', () => {
  assert.equal(toLatex('X → Y'), 'X \\to Y')
  assert.equal(toLatex('a_b 5% #1'), 'a\\_b 5\\% \\#1')
  assert.equal(toLatex('Z = X W'), 'Z = X W')
})

test('real Kipf plan, step 2: the formula spans are found, the words around them stay text', () => {
  const step = planItems(KIPF, 3).steps[1]
  const segs = splitFormulas(step)
  const math = segs.filter((s) => s.kind === 'math')
  assert.deepEqual(math.map((m) => m.latex), ['\\tilde{A} = A + I', '\\hat{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}'])
  assert.ok(segs[0].kind === 'text' && segs[0].text === 'Propagation: ')
  // nothing lost or invented: put the text back together
  assert.equal(segs.map((s) => s.text).join(''), step)
  assert.ok(segs.some((s) => s.kind === 'text' && s.text.includes('(renormalization trick, Eq. 8)')))
})

test('real Kipf plan: other steps without ^ or accents contain no formula (lr=0.01 and Train=140 stay text)', () => {
  const all = planItems(KIPF, 20, 200).steps
  for (const s of all) {
    const m = splitFormulas(s).filter((x) => x.kind === 'math')
    if (!/\^|[\u0302\u0303\u0304]|[ÃÂ]/.test(s.normalize('NFD'))) assert.equal(m.length, 0, s)
  }
  assert.equal(splitFormulas('Adam, lr=0.01, full-batch').every((x) => x.kind === 'text'), true)
})

test('$...$ is taken as written, also next to plain text', () => {
  const segs = splitFormulas('The loss is $\\mathcal{L} = -\\sum y \\log p$ on the train set')
  assert.deepEqual(segs.map((s) => s.kind), ['text', 'math', 'text'])
  assert.equal((segs[1] as { latex: string }).latex, '\\mathcal{L} = -\\sum y \\log p')
})

test('text without formulas comes back as one text segment; join is always lossless', () => {
  for (const t of ['no formula here', 'a b^2 c, d', 'X̂ = Y; next']) assert.equal(splitFormulas(t).map((s) => s.text).join(''), t)
  assert.deepEqual(splitFormulas('no formula here'), [{ kind: 'text', text: 'no formula here' }])
})

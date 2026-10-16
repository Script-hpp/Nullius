import test from 'node:test'
import assert from 'node:assert/strict'
import { latexRows, looksLikeLatexRow, matchesTarget } from './latexRow.ts'

test('the layout row from the discover page reads as its cells', () => {
  const q = '\\hline \\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{ETTh1}}} & 96 & \\textbf{0.381} & 0.405 & 0.417'
  assert.deepEqual(latexRows(q), [['ETTh1', '96', '0.381', '0.405', '0.417']])
})

test('real quotes from the exported runs', () => {
  assert.deepEqual(latexRows('\\multirow{2}{*}{SparseTSF (ours)} & \\textbf{0.359} & \\textbf{0.397} & \\textbf{0.404} & \\textbf{0.417}'),
    [['SparseTSF (ours)', '0.359', '0.397', '0.404', '0.417']])
  const fits = latexRows('FITS & \\textbf{0.372} & \\textbf{0.404} & 0.303 & {\\underline{0.366}} & \\textbf{0.415}')
  assert.deepEqual(fits, [['FITS', '0.372', '0.404', '0.303', '0.366', '0.415']])
  assert.deepEqual(latexRows('\\midrule\n\\textbf{Cora}       & 80.82±0.14 & 81.40±0.29 & {\\ul 83.72±0.52} & \\textbf{84.23±0.62*}'),
    [['Cora', '80.82±0.14', '81.40±0.29', '83.72±0.52', '84.23±0.62*']])
})

test('row ends split rows; escaped ampersands, math marks and unknown commands do not leak', () => {
  assert.deepEqual(latexRows('A & $0.5 \\pm 0.1$ \\\\ \\hline B \\& C & \\cellcolor{gray}0.7 \\\\'), [['A', '0.5 ± 0.1'], ['B & C', '0.7']])
})

test('plain text and markdown tables are not LaTeX rows', () => {
  assert.equal(looksLikeLatexRow('GCN achieves 81.5% on Cora'), false)
  assert.equal(looksLikeLatexRow('| GCN | 81.5 |'), false)
  assert.equal(looksLikeLatexRow('Master\'s degree in Telecommuniation \\& Networks'), false)
  assert.deepEqual(latexRows('GCN achieves 81.5% on Cora'), [])
  assert.deepEqual(latexRows(undefined), [])
})

test('no command text is left in any cell of the exported quotes', () => {
  const rows = latexRows('\\hline \\parbox[t]{2mm}{\\multirow{4}{*}{\\rotatebox[origin=c]{90}{ETTh1}}} & 96 & \\textbf{0.381} \\\\ \\cline{2-4} & 192 & \\underline{0.4}')
  for (const r of rows) for (const c of r) assert.ok(!/[\\{}]/.test(c), c)
})

test('the claimed value is found whether the cell holds 0.372 or 83.72±0.52', () => {
  assert.ok(matchesTarget('0.372', 0.372))
  assert.ok(matchesTarget('83.72±0.52', 0.8372))
  assert.ok(!matchesTarget('96', 0.372))
  assert.ok(!matchesTarget('ETTh1', 0.372))
})

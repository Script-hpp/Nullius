import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { targetSource, setupSourceUrl, sourceUrl } from './targetSource.ts'

const glinear = JSON.parse(readFileSync(new URL('../public/data/disc_2501_01087/20261010-161916/dossier.json', import.meta.url), 'utf8'))

test('GLinear exposes its recorded original quote without upgrading failed verification', () => {
  const before = JSON.stringify(glinear)
  const source = targetSource(glinear.claims[0], glinear.sources)
  assert.equal(source.url, 'https://arxiv.org/html/2501.01087v1')
  assert.match(source.quote!, /TABLE II/)
  assert.equal(source.extracted, false)
  assert.equal(source.columnVerified, false)
  assert.equal(JSON.stringify(glinear), before)
})

test('an extracted target with unknown table column remains unverified', () => {
  const source = targetSource({ source: 'Table 3, arXiv v5 (pdf)', target_evidence: { url: 'https://arxiv.org/pdf/2303.06053v5', quote: 'ETTh1 96 .361 .392', table: 'Table 3', column_check: 'nicht pruefbar' } })
  assert.equal(source.url, 'https://arxiv.org/pdf/2303.06053v5')
  assert.equal(source.recordedLocation, 'Table 3, arXiv v5 (pdf)')
  assert.equal(source.columnVerified, false)
  assert.equal(targetSource({ target_evidence: { column_check: 'ok' } }).columnVerified, true)
})

test('safe source fallbacks support old exports and reject executable URLs', () => {
  assert.equal(sourceUrl('javascript:alert(1)'), null)
  assert.equal(targetSource({ target_evidence: { url: 'javascript:alert(1)' } }, [{ kind: 'paper', url: 'https://arxiv.org/abs/1234.56789' }]).url, 'https://arxiv.org/abs/1234.56789')
  assert.equal(targetSource({ evidence_urls: ['https://example.com/x', 'https://arxiv.org/pdf/1234.56789'] }).url, 'https://arxiv.org/pdf/1234.56789')
  assert.equal(targetSource({}).url, null)
})

test('setup code references point at the recorded commit and preserve provenance', () => {
  const dossier = { repo: { url: 'https://github.com/t-rizvi/GLinear.git' } }
  assert.equal(setupSourceUrl('repo:models/GLinear.py@f54bc96', dossier, {}), 'https://github.com/t-rizvi/GLinear/blob/f54bc96/models/GLinear.py')
  assert.equal(setupSourceUrl('unknown source', dossier, {}), null)
  assert.equal(setupSourceUrl('repo:../x.py@f54bc96', dossier, {}), 'https://github.com/t-rizvi/GLinear.git')
})

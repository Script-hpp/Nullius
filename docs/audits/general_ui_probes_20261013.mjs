// Pure event-model probes, no browser, HTTP, model or paid job.
// node docs/audits/general_ui_probes_20261013.mjs
import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { buildRunModel } from '../../web/src/components/run/phases.ts'

const stopped = buildRunModel([
  { ts: 1, type: 'run_start', seeds: 3 },
  { ts: 2, type: 'claim_start', claim_id: 'a', metric: 'mse', target: .371, tolerance: .01 },
  { ts: 3, type: 'code_written', candidate: 0, code_hash: 'a' },
  { ts: 4, type: 'error', message: 'Budget exhausted' },
], { running: false })
assert.equal(stopped.candidateSelectionPending, true)
assert.equal(stopped.stats, null)

const mixed = buildRunModel([
  { ts: 1, type: 'run_start', seeds: 1 },
  { ts: 2, type: 'claim_start', claim_id: 'classification', text: 'Graph accuracy', metric: 'accuracy', target: .83, tolerance: .01 },
  { ts: 3, type: 'attempt', claim_id: 'classification', seed: 0, code_hash: 'graph', measured: .83 },
  { ts: 4, type: 'measurements_selected', claim_id: 'classification', attempts: [{ seed: 0, code_hash: 'graph', measured: .83 }] },
  { ts: 5, type: 'verdict', claim_id: 'classification', mean: .83, std: 0 },
  { ts: 6, type: 'claim_start', claim_id: 'forecast', text: 'Forecast MSE', metric: 'mse', target: .371, tolerance: .01 },
  { ts: 7, type: 'attempt', claim_id: 'forecast', seed: 0, code_hash: 'forecast', measured: .4 },
  { ts: 8, type: 'measurements_selected', claim_id: 'forecast', attempts: [{ seed: 0, code_hash: 'forecast', measured: .4 }] },
  { ts: 9, type: 'verdict', claim_id: 'forecast', mean: .4, std: 0 },
], { running: false })
assert.equal(mixed.claimText, 'Graph accuracy')
assert.equal(mixed.metric, 'accuracy')
assert.equal(mixed.target, .83)
assert.equal(mixed.stats.mean, .4)
assert.equal(mixed.verdict.mean, .4)

const savedDir = new URL('../../web/public/data/kipf_gcn_2017/20261002-110746/', import.meta.url)
const savedDossier = JSON.parse(readFileSync(new URL('dossier.json', savedDir), 'utf8'))
const savedEvents = readFileSync(new URL('events.jsonl', savedDir), 'utf8').trim().split('\n').map(JSON.parse)
const saved = buildRunModel(savedEvents, { running: false })
const paper = savedDossier.claims.find(c => c.origin === 'paper')
const paperVerdict = savedDossier.verdicts.find(v => v.claim_id === paper.id)
assert.equal(paperVerdict.urteil, 'reproduziert')
assert.equal(saved.target, paper.target)
assert.equal(saved.verdict.urteil, 'nicht reproduziert')
assert.equal(saved.verdict.claim_id, 'gcn_cora_acc_placebo')

const result = {
  confirmed_probes: 3,
  results: [
    { probe: 'stopped_candidate_status', confirmed: true, observed: { running: false, candidateSelectionPending: stopped.candidateSelectionPending, stats: stopped.stats } },
    { probe: 'multi_claim_event_mix', confirmed: true, observed: { claimText: mixed.claimText, metric: mixed.metric, target: mixed.target, chartMean: mixed.stats.mean, verdictMean: mixed.verdict.mean } },
    { probe: 'existing_saved_placebo_verdict_mix', confirmed: true, observed: { run: 'kipf_gcn_2017--20261002-110746', paperTarget: paper.target, paperVerdict: paperVerdict.urteil, renderedTarget: saved.target, renderedVerdict: saved.verdict.urteil, renderedVerdictClaim: saved.verdict.claim_id } },
  ],
}
writeFileSync(new URL('./general_ui_probes_20261013.json', import.meta.url), JSON.stringify(result, null, 2) + '\n')
console.log(JSON.stringify(result))

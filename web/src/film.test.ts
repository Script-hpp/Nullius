import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { FILM_MS, addressBar, communityScene, isPaperCall, cardWeight, fitSteps, mappingComments, definitionLine, pipelineOf, planCodeLinks, splitCategory, branchX, branches, claimEvidence, codeLines, diffFlash, dossierAt, durationText, measurements, planItems, mathWords, stripMarkdown, presentScenes, repairScene, sceneAt, scenePlan, sourceDomains, tavilySentence, typed, type SceneId } from './film.ts'
import type { Event } from './data.ts'

const all: SceneId[] = ['tavily', 'code', 'repair', 'sandbox', 'harness', 'verdict']
const events = JSON.parse(readFileSync(new URL('./fixtures/nlinear_story_events.json', import.meta.url), 'utf8')) as Event[]

test('plan: always 30 s, scenes follow each other without gaps, also without the repair scene', () => {
  for (const ids of [all, all.filter((s) => s !== 'repair'), ['harness', 'verdict'] as SceneId[]]) {
    const plan = scenePlan(new Set(ids))
    assert.equal(plan[0].start, 0)
    assert.equal(plan[plan.length - 1].end, FILM_MS)
    plan.slice(1).forEach((s, i) => assert.equal(s.start, plan[i].end))
    assert.deepEqual(plan.map((s) => s.id), ids)
  }
})

test('plan: the repair scene shortens nothing it does not have to, the others stretch to fill 30 s', () => {
  const with_ = scenePlan(new Set(all)); const without = scenePlan(new Set(all.filter((s) => s !== 'repair')))
  assert.ok(without[0].end - without[0].start > with_[0].end - with_[0].start)
})

test('sceneAt: index and progress, clamped at both ends', () => {
  const plan = scenePlan(new Set(all))
  assert.deepEqual(sceneAt(plan, 0), { index: 0, p: 0 })
  assert.equal(sceneAt(plan, FILM_MS + 99).index, plan.length - 1)
  assert.equal(sceneAt(plan, FILM_MS).p, 1)
  const mid = (plan[2].start + plan[2].end) / 2
  assert.deepEqual([sceneAt(plan, mid).index, Math.round(sceneAt(plan, mid).p * 100)], [2, 50])
  assert.deepEqual(sceneAt([], 5), { index: 0, p: 0 })
})

test('claim evidence: the real quote, table and page; nothing without a quote', () => {
  const c = { target_evidence: { quote: 'GCN (this paper) 70 .3 (7s) 81 .5 (4s)', table: 'Table 2', url: 'https://arxiv.org/pdf/1609.02907', arxiv_version: 'v4' } }
  assert.deepEqual(claimEvidence(c), { quote: 'GCN (this paper) 70 .3 (7s) 81 .5 (4s)', table: 'Table 2', url: 'https://arxiv.org/pdf/1609.02907', version: 'v4' })
  assert.equal(claimEvidence({ target_evidence: null }), null)
  assert.equal(addressBar('https://www.arxiv.org/pdf/1609.02907'), 'arxiv.org/pdf/1609.02907')
})

test('source domains: the evidence page first, then Tavily targets, each once', () => {
  const ev = [{ ts: 0, type: 'tavily_call', target: 'https://github.com/tkipf/gcn' }, { ts: 1, type: 'tavily_call', target: 'https://arxiv.org/pdf/1609.02907' }, { ts: 2, type: 'tavily_call', target: 'not a url' }] as Event[]
  assert.deepEqual(sourceDomains(ev, 'https://arxiv.org/pdf/1609.02907'), ['arxiv.org', 'github.com'])
  assert.deepEqual(sourceDomains([], undefined), [])
})

test('code lines: no blanks or comment-only lines, limited, long lines cut', () => {
  const code = '# plan comment\n\nimport numpy as np\n\nx = 1  # note\n' + 'y = 2\n'.repeat(30) + 'z = "' + 'a'.repeat(200) + '"\n'
  const l = codeLines(code, 5)
  assert.equal(l.length, 5)
  assert.equal(l[0], 'import numpy as np')
  assert.ok(!l.some((x) => x.startsWith('#')))
  assert.ok(codeLines('z = "' + 'a'.repeat(200) + '"')[0].length <= 78)
})

const KIPF_PLAN = readFileSync(new URL('../public/data/kipf_gcn_2017/20260929-130133/plan.md', import.meta.url), 'utf8')

test('plan items on the real Kipf plan: the bold first line is the title, the list items are the steps', () => {
  const { title, steps } = planItems(KIPF_PLAN, 3)
  assert.equal(title, 'Reproduction Plan: GCN on Cora (Planetoid split)')
  assert.equal(steps.length, 3)
  assert.equal(steps[0], 'Model: 2-layer GCN: X → GCN(16, ReLU, dropout) → GCN(7, linear, dropout) → softmax')
  assert.ok(steps[1].startsWith('Propagation: Ã = A + I; Â = D̃^(-1/2) Ã D̃^(-1/2)'))
  assert.ok(steps.every((l) => !/[*`#]/.test(l) && !/^\d+[.)]/.test(l)))
})

test('plan items: heading line is a title, a plan without a list falls back to longer lines, nothing else is dropped', () => {
  assert.deepEqual(planItems('# Plan for X\n\n- Use a linear layer with 96 outputs\n- Train with Adam\n', 3), { title: 'Plan for X', steps: ['Use a linear layer with 96 outputs', 'Train with Adam'], numbers: [1, 2] })
  assert.deepEqual(planItems('First we normalise the data per channel.\nThen a linear model is fitted.\nok', 3), { title: null, steps: ['First we normalise the data per channel.', 'Then a linear model is fitted.'], numbers: [1, 2] })
  assert.equal(planItems('', 3).steps.length, 0)
})

test('plan items: long lines are cut at a space, never in the middle of a word', () => {
  const long = '1. ' + 'word '.repeat(40)
  const [l] = planItems(long, 1, 50).steps
  assert.ok(l.endsWith('…') && l.length <= 51 && !/wor…$/.test(l))
})

test('stripMarkdown removes markers only', () => {
  assert.equal(stripMarkdown('  - **Loss**: `MSE` on 96 steps'), 'Loss: MSE on 96 steps')
  assert.equal(stripMarkdown('## Heading'), 'Heading')
})

test('math words: superscripts for ^(...) and ^x, formulas stay one piece, nothing else is interpreted', () => {
  const w = mathWords('Â = D̃^(-1/2) Ã D̃^(-1/2) and x^2 or 2^-3')
  const formulas = w.filter((x) => x.formula)
  assert.equal(formulas.length, 4)
  assert.deepEqual(formulas[0].parts, [{ kind: 'text', text: 'D̃' }, { kind: 'sup', text: '-1/2' }])
  assert.deepEqual(formulas[2].parts, [{ kind: 'text', text: 'x' }, { kind: 'sup', text: '2' }])
  assert.deepEqual(formulas[3].parts, [{ kind: 'text', text: '2' }, { kind: 'sup', text: '-3' }])
  assert.equal(w.map((x) => x.parts.map((p) => p.text).join('')).join('').replace(/\s+/g, ' ').includes('and'), true)
  assert.equal(mathWords('no formula here').every((x) => !x.formula), true)
})

test('typed: a prefix of the text, clamped', () => {
  assert.equal(typed('abcdef', 0.5), 'abc')
  assert.equal(typed('abcdef', 2), 'abcdef')
  assert.equal(typed('abcdef', -1), '')
})

test('branches: one per seed, a valid measurement wins over an earlier failure; measurements keep event order', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 1, measured: null }, { ts: 1, type: 'attempt', claim_id: 'c', seed: 0, measured: 0.8 },
    { ts: 2, type: 'attempt', claim_id: 'c', seed: 1, measured: 0.7 }, { ts: 3, type: 'attempt', claim_id: 'c', seed: 2, measured: null },
  ] as Event[]
  assert.deepEqual(branches(ev), [{ seed: 0, ok: true }, { seed: 1, ok: true }, { seed: 2, ok: false }])
  assert.deepEqual(measurements(ev, 'c'), [0.8, 0.7])
  assert.deepEqual([branchX(0, 1), branchX(0, 3), branchX(2, 3)], [0.5, 0.06, 0.94])
})

test('dossierAt: the first k measurements, the verdict only when asked', () => {
  const ev = [0.8, 0.82, 0.81].map((m, i) => ({ ts: i, type: 'attempt', claim_id: 'c', seed: i, measured: m })) as Event[]
  const d = { claims: [{ id: 'c' }], attempts: [], verdicts: [{ claim_id: 'c' }] }
  assert.equal(dossierAt(d, 'c', ev, 2, false).attempts.length, 2)
  assert.deepEqual(dossierAt(d, 'c', ev, 2, false).verdicts, [])
  assert.equal(dossierAt(d, 'c', ev, 3, true).verdicts.length, 1)
})

test('repair scene on a real run: first repair card, a sentence that is true for it, the version counter', () => {
  const r = repairScene(events, 'nlinear_etth1_96_mse')!
  assert.equal(r.card.source, 'error')
  assert.equal(r.sentence, 'The code failed. The agent reads the error, searches the web and fixes it.')
  assert.deepEqual(r.version, { from: 1, to: 2 })
  assert.equal(repairScene(events, 'other'), null)
})

test('repair sentence without a search does not mention one', () => {
  const ev = [
    { ts: 0, type: 'attempt', claim_id: 'c', seed: 0, n: 1, code_hash: 'a', status: 'fehler', measured: null, error: 'ValueError: x' },
    { ts: 1, type: 'repair', claim_id: 'c', seed: 0, from_hash: 'a', error: 'ValueError: x' },
  ] as Event[]
  assert.equal(repairScene(ev, 'c')!.sentence, 'The code failed. The agent reads the error and fixes it.')
})

test('diff flash: only changed lines, limited', () => {
  const f = diffFlash('a\nb\nc\nd', 'a\nB\nc\nd\ne\nf\ng', 4)
  assert.deepEqual(f.map((l) => l.t), ['del', 'add', 'add', 'add'])
  assert.deepEqual(diffFlash('x', 'x'), [])
})

test('present scenes: a scene appears only if its data exists', () => {
  const claim = { id: 'c', target_evidence: null }
  const none = presentScenes({ events: [], claim, hasPlan: false, hasCode: false, hasVerdict: false, repair: false })
  assert.equal(none.size, 0)
  const ev = [{ ts: 0, type: 'tavily_call', target: 'https://x.org/a' }, { ts: 1, type: 'attempt', claim_id: 'c', seed: 0, measured: 0.8 }] as Event[]
  const some = presentScenes({ events: ev, claim, hasPlan: true, hasCode: false, hasVerdict: true, repair: false })
  assert.deepEqual([...some].sort(), ['code', 'harness', 'sandbox', 'tavily', 'verdict'])
})

test('duration text', () => {
  const e = (t: number) => ({ ts: t, type: 'x' }) as Event
  assert.equal(durationText([e(0), e(41 * 60)]), '41 minutes')
  assert.equal(durationText([e(0)]), null)
})

test('first sentence only claims a found number when there is evidence', () => {
  assert.match(tavilySentence(true), /finds the number it reports/)
  assert.doesNotMatch(tavilySentence(false), /number/)
})

test('category split: text before the first colon, only if short and plain', () => {
  assert.deepEqual(splitCategory('Model: 2-layer GCN: X → GCN(16)'), { category: 'Model', body: '2-layer GCN: X → GCN(16)' })
  assert.deepEqual(splitCategory('Lookback / Horizon: 336 / 96, 7 channels'), { category: 'Lookback / Horizon', body: '336 / 96, 7 channels' })
  assert.deepEqual(splitCategory('Row-normalize the vectors'), { category: null, body: 'Row-normalize the vectors' })
  assert.equal(splitCategory('Z = f(x): then more').category, null)   // a formula is not a category
  const real = planItems(KIPF_PLAN, 4).steps.map((s) => splitCategory(s).category)
  assert.deepEqual(real, ['Model', 'Propagation', 'Features', 'Adjacency'])
})

test('pipeline: arrow chains become nodes, arrows inside brackets do not split, prose is not a pipeline', () => {
  const p = pipelineOf('2-layer GCN: X → GCN(16, ReLU, dropout) → GCN(7, linear, dropout) → softmax')!
  assert.equal(p.lead, '2-layer GCN:')
  assert.deepEqual(p.nodes, ['X', 'GCN(16, ReLU, dropout)', 'GCN(7, linear, dropout)', 'softmax'])
  assert.deepEqual(pipelineOf('f(a -> b) -> g')!.nodes, ['f(a -> b)', 'g'])
  assert.equal(pipelineOf('Adam, lr=0.01, full-batch gradient descent'), null)
  assert.equal(pipelineOf('We first normalise every channel of the input and then, much later in the long pipeline of steps → done'), null)
  assert.equal(pipelineOf('A → B')!.lead, '')
})

const NLINEAR_CODE = `import torch
# Plan step -> implemented in: function or class name
# 1. Model -> NLinear class
# 2. Lookback / Horizon -> constants LOOKBACK, HORIZON
# 3. Preprocessing -> normalize_window, denormalize_prediction functions
# 4. Training data -> create_windows function
# 9. Seeding -> missing_function function
LOOKBACK = 336
HORIZON = 96
class NLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(LOOKBACK, HORIZON)

def normalize_window(x):
    last = x[-1:]
    return x - last, last
`

const steps5 = [['Model', 1], ['Lookback / Horizon', 2], ['Preprocessing', 3], ['Training data', 4], ['Seeding', 5]].map(([category, number]) => ({ category: category as string, number: number as number }))

test('plan <-> code links: only comments whose category matches and whose name is really defined', () => {
  const links = planCodeLinks(NLINEAR_CODE, steps5)
  assert.deepEqual(links.map((l) => [l.step, l.name]), [[0, 'NLinear'], [1, 'LOOKBACK'], [2, 'normalize_window']])   // create_windows is not defined, missing_function neither
  assert.equal(links[0].lines[0], 'class NLinear(torch.nn.Module):')
  assert.equal(links[1].lines[0], 'LOOKBACK = 336')
  assert.deepEqual(planCodeLinks(NLINEAR_CODE, [{ category: 'Optimizer', number: 9 }, { category: null, number: 10 }]), [])   // no matching category: no link
  assert.deepEqual(planCodeLinks('import os\nx = 1\n', [{ category: 'Model', number: 1 }]), [])                                // no mapping comments: no link
})

const BY_NUMBER_CODE = `# Plan step 1 -> implemented in: NLinear class
# Plan step 2 -> implemented in: LOOKBACK, HORIZON variables
# Plan step 3 -> implemented in: normalize_last_value / denormalize_last_value
# Plan step 4 -> implemented in: not_there function

LOOKBACK = 336
class NLinear:
    pass
def normalize_last_value(x):
    return x
`

test('plan <-> code links by step number ("Plan step 3 -> implemented in: ..."), using the numbers written in the plan', () => {
  const links = planCodeLinks(BY_NUMBER_CODE, [{ category: 'Model', number: 1 }, { category: 'Whatever', number: 3 }, { category: 'Other', number: 4 }])
  assert.deepEqual(links.map((l) => [l.step, l.name]), [[0, 'NLinear'], [1, 'normalize_last_value']])   // step 4: not_there is not defined, no link
  assert.deepEqual(planItems('1. **A**: x is long enough\n3. **B**: y is long enough\n', 5).numbers, [1, 3])
})

test('mapping comments are read from the top of the code only', () => {
  assert.deepEqual(mappingComments(NLINEAR_CODE).map((m) => m.category), ['Model', 'Lookback / Horizon', 'Preprocessing', 'Training data', 'Seeding'])
  assert.deepEqual(mappingComments(BY_NUMBER_CODE).map((m) => m.number), [1, 2, 3, 4])
  assert.equal(mappingComments('x = 1\n'.repeat(100) + '# 1. Model -> X\n').length, 0)
  assert.equal(definitionLine(NLINEAR_CODE, 'HORIZON'), 8)
})

test('fit steps: the first steps that fit the budget, at least one, formulas weigh more than plain text', () => {
  const steps = planItems(KIPF_PLAN, 6).steps
  assert.ok(cardWeight(steps[1]) > cardWeight(steps[2]))          // formula card > plain card
  assert.ok(cardWeight(steps[0]) > cardWeight(steps[2]))          // long arrow chain > plain card
  assert.deepEqual(fitSteps(steps, 5.2), steps.slice(0, 3))
  assert.deepEqual(fitSteps(steps, 3.2), steps.slice(0, 1))
  assert.deepEqual(fitSteps(steps, 0.1), steps.slice(0, 1))       // never empty
  assert.deepEqual(fitSteps([], 5), [])
  assert.equal(fitSteps(['Plain: a', 'Plain: b', 'Plain: c', 'Plain: d'], 3.2).length, 3)
})

test('community scene: only for runs with a community check; sentence matches the number of notes; paper calls exclude the discussion search', () => {
  assert.equal(communityScene({}), null)
  const none = communityScene({ community: { findings: [] } })!
  assert.match(none.sentence, /found nothing\. It never changes the verdict/)
  const c = communityScene({ community: { findings: [
    { statement: 'Seed matters', quote: 'x'.repeat(200), url: 'https://github.com/a/b/issues/1', kind: 'protocol note' },
    { statement: 'Second', quote: 'q', url: 'https://www.reddit.com/r/x', kind: 'other' },
    { statement: 'Third', quote: 'q', url: 'https://x.org', kind: 'other' },
  ] } })!
  assert.equal(c.total, 3)
  assert.equal(c.notes.length, 2)
  assert.equal(c.notes[0].host, 'github.com')
  assert.ok(c.notes[0].quote.length <= 150 && c.notes[0].quote.endsWith('…'))
  assert.match(c.sentence, /3 notes found, each with a quote from the page\. They never change the verdict/)
  assert.equal(communityScene({ community: { findings: [{ statement: 's', quote: 'q', url: 'https://a.org' }] } })!.sentence.includes('1 note found'), true)
  const calls = [{ ts: 0, type: 'tavily_call', role: 2, target: 'https://arxiv.org/abs/1' }, { ts: 1, type: 'tavily_call', role: 3, target: 'https://reddit.com/x' }, { ts: 2, type: 'tavily_call', role: 6, target: 'https://x.org' }] as Event[]
  assert.deepEqual(calls.map(isPaperCall), [true, false, false])
  assert.deepEqual(sourceDomains(calls), ['arxiv.org'])
  const claim = { id: 'c', target_evidence: null }
  const scenes = presentScenes({ events: [], claim, hasPlan: true, hasCode: false, hasVerdict: true, repair: false, community: true })
  assert.deepEqual([...scenes].sort(), ['code', 'community', 'verdict'])
  assert.deepEqual(scenePlan(scenes).map((x) => x.id), ['code', 'community', 'verdict'])   // the stamp stays the last scene
})

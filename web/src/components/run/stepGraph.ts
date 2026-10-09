import type { Event } from '../../data'
import type { Tone } from '../../labels'
import { VERDICT_SENTENCE, translateClaim } from '../../labels'
import { fmtSeconds, fmtValue, type Expand, type RunModel } from './phases'

// Turns the event list of a run into a flow graph: nodes (steps) and edges (data flow, repair loops).
// Pure: every value comes from an event; a missing value drops the line. Layout is in grid units, the component places the pixels.

export type StepKind = 'paper' | 'src_paper' | 'src_readme' | 'src_repo' | 'src_error' | 'claim' | 'plan' | 'code' | 'seed' | 'harness' | 'verdict'
export type StepState = 'pending' | 'active' | 'done' | 'failed'
export type Fact = { label: string; value: string }

export type StepNode = {
  id: string
  kind: StepKind
  title: string
  sub: string | null
  state: StepState
  tone: Tone
  what: string            // "What happens here", plain language
  why: string             // "Why this matters", plain language
  facts: Fact[]           // real data of the run
  expand?: Expand
  row: number             // grid row
  col: number             // position within the row, 0-based
  rowSize: number         // number of nodes in the row (for centering)
}

export type StepEdge = { id: string; source: string; target: string; loop?: 'left' | 'right'; label?: string; dashed?: boolean }
export type StepGraph = { nodes: StepNode[]; edges: StepEdge[] }

const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : x != null && x !== '' && Number.isFinite(Number(x)) ? Number(x) : null)
const host = (url: string) => { try { const u = new URL(url); return u.host + u.pathname.replace(/\/$/, '') } catch { return url } }
const clip = (s: string, n = 160) => (s.length > n ? s.slice(0, n - 1) + '…' : s)
const lastLine = (s: unknown) => String(s ?? '').split('\n').map((x) => x.trim()).filter(Boolean).pop() ?? ''
const GROUP_FROM = 4   // from this many seeds on, the seeds are one node

export function buildStepGraph(events: Event[], model: RunModel, running: boolean): StepGraph {
  const of = (t: string) => events.filter((e) => e.type === t)
  const has = (t: string) => events.some((e) => e.type === t)
  const metric = model.metric
  const runStart = events.find((e) => e.type === 'run_start')
  const verdictEv = model.verdict
  const verdictSeen = !!verdictEv
  const active = (phase: string) => model.phases.find((p) => p.id === phase)?.state === 'active'
  const st = (done: boolean, isActive: boolean, failed = false): StepState => (failed ? 'failed' : done ? 'done' : isActive && running ? 'active' : 'pending')

  // --- Tavily sources ---
  const paperSrc = events.find((e) => e.type === 'source' && e.kind === 'paper')
  const paperCall = events.find((e) => e.type === 'tavily_call' && Number(e.role) === 2)
  const readmeSrc = events.find((e) => e.type === 'source' && e.kind === 'readme')
  const readmeCall = events.find((e) => e.type === 'tavily_call' && Number(e.role) === 4)
  const repoSearch = events.find((e) => e.type === 'repo_search')
  const errorSearches = of('error_search')

  const srcFacts = (src: Event | undefined, call: Event | undefined): Fact[] => {
    const f: Fact[] = []
    const url = String(src?.url ?? call?.target ?? '')
    if (url) f.push({ label: 'Page', value: host(url) })
    if (src) {
      f.push({ label: 'How', value: src.mode === 'live' ? 'Live Tavily call at run time' : 'Saved copy of an earlier Tavily call' })
      const chars = num(src.n_chars)
      if (chars != null) f.push({ label: 'Read', value: `${chars.toLocaleString('en-US')} characters` })
      if (src.retrieved_at) f.push({ label: 'Retrieved', value: String(src.retrieved_at).replace('T', ' ').slice(0, 16) + ' UTC' })
      if (src.sha256) f.push({ label: 'Fingerprint (SHA-256)', value: String(src.sha256).slice(0, 16) + '…' })
    }
    return f
  }
  const srcSub = (src: Event | undefined, call: Event | undefined, later: boolean) =>
    src ? (src.mode === 'live' ? 'live Tavily call' : 'saved copy') : call ? (running && !later ? 'reading…' : 'fetched') : later ? 'not recorded in this run' : null

  // --- claim ---
  const target = events.find((e) => e.type === 'target_resolved')
  const check = events.find((e) => e.type === 'target_check')
  const claimFacts: Fact[] = []
  if (model.claimText) claimFacts.push({ label: 'Claim', value: translateClaim(model.claimText) })
  if (target) {
    claimFacts.push({ label: 'Target value', value: fmtValue(metric, num(target.value), 2) })
    if (target.table) claimFacts.push({ label: 'Found in', value: String(target.table) })
    claimFacts.push({ label: 'Table column', value: target.column_check === 'ok' ? 'checked: the number belongs to the proposed method' : 'could not be checked' })
  }
  if (check) claimFacts.push({ label: 'Second reading', value: check.agrees === true ? 'agrees with the first' : 'differs from the first' })
  if (model.tolerance != null) claimFacts.push({ label: 'Tolerance', value: `± ${fmtValue(metric, model.tolerance, 2)}, fixed before the run` })

  // --- repairs (loops back to the code) ---
  const crash = of('repair').length
  const rewrite = of('rewrite').length
  const faultSeed = of('fault_repair').filter((e) => e.runde !== 'mittelwert').length
  const faultMean = of('fault_repair').filter((e) => e.runde === 'mittelwert').length
  const graphFix = of('graph_repair').length
  const seedLoop = [crash && `crash ×${crash}`, rewrite && `new code ×${rewrite}`, faultSeed && `baseline fix ×${faultSeed}`, graphFix && `graph fix ×${graphFix}`].filter(Boolean).join(', ')

  // --- setup / plan / code ---
  const setup = events.find((e) => e.type === 'setup_details')
  const planEv = events.find((e) => e.type === 'plan')
  const planRow = model.phases.flatMap((p) => p.rows).find((r) => r.expand?.kind === 'plan')
  const codeRows = model.phases.flatMap((p) => p.rows).filter((r) => r.expand?.kind === 'code')
  const lastCode = codeRows[codeRows.length - 1]
  const codeVersions = new Set(codeRows.map((r) => (r.expand as Extract<Expand, { kind: 'code' }>).hash)).size
  const codeMissing = of('code_missing').length
  const codeFacts: Fact[] = []
  if (codeVersions) codeFacts.push({ label: 'Code versions', value: String(codeVersions) })
  if (seedLoop) codeFacts.push({ label: 'Repairs after seed runs', value: seedLoop })
  if (faultMean) codeFacts.push({ label: 'Fix after all seeds', value: `${faultMean}x` })
  if (codeMissing) codeFacts.push({ label: 'Answers without code', value: `${codeMissing}; the model was asked again` })
  const planRetries = of('plan_retry').length
  const planFacts: Fact[] = planRetries ? [{ label: 'Asked again', value: `${planRetries}x because the first answer was not a short plan` }] : []

  // --- seeds ---
  const seedAttempts = new Map<number, Event[]>()
  for (const e of of('attempt')) {
    const s = num(e.seed) ?? 0
    seedAttempts.set(s, [...(seedAttempts.get(s) ?? []), e])
  }
  const maxSeen = seedAttempts.size ? Math.max(...seedAttempts.keys()) + 1 : 0
  const seedCount = Math.max(model.seedsTotal ?? 0, maxSeen, 1)
  const validBySeed = new Map(model.attempts.map((a) => [a.seed, a]))
  const seed0Done = validBySeed.has(0)
  const singles: StepNode[] = []
  for (let s = 0; s < seedCount; s++) {
    const list = seedAttempts.get(s) ?? []
    const valid = validBySeed.get(s)
    const lastTry = list[list.length - 1]
    const failed = !valid && !!lastTry && num(lastTry.measured) == null && !running
    const isNext = s === 0 ? !seed0Done : seed0Done
    const facts: Fact[] = []
    if (valid) facts.push({ label: 'Harness score', value: fmtValue(metric, valid.measured) })
    if (valid?.seconds != null) facts.push({ label: 'Training time', value: fmtSeconds(valid.seconds) })
    if (list.length) facts.push({ label: 'Attempts', value: String(list.length) })
    if (!valid && lastTry?.error) facts.push({ label: 'Last error', value: clip(lastLine(lastTry.error), 200) })
    singles.push({
      id: `seed_${s}`, kind: 'seed', title: `Seed ${s}`, sub: valid ? fmtValue(metric, valid.measured) : failed ? 'no valid run' : list.length ? 'retrying…' : null,
      state: st(!!valid, isNext && has('code_written') && !verdictSeen, failed), tone: valid ? 'gray' : failed ? 'red' : 'gray',
      what: s === 0
        ? 'The code trains in its own sandbox branch. It must write predictions only; it never sees the test labels. This first run catches crashes early.'
        : 'The same code runs again with a different random start, in its own sandbox branch, in parallel with the other seeds.',
      why: 'One run can be lucky or unlucky. Several seeds show how much the result changes by chance. The verdict uses their mean.',
      facts, row: 5, col: s, rowSize: seedCount,
    })
  }
  let seedNodes = singles
  if (seedCount >= GROUP_FROM) {
    const scores = model.attempts.map((a) => a.measured)
    const mean = scores.length ? scores.reduce((t, x) => t + x, 0) / scores.length : null
    const anyActive = singles.some((n) => n.state === 'active')
    const group: StepNode = {
      id: 'seeds', kind: 'seed', title: `${seedCount} seeds`,
      sub: mean != null ? `mean ${fmtValue(metric, mean)} · range ${fmtValue(metric, Math.min(...scores))}–${fmtValue(metric, Math.max(...scores))}${scores.length < seedCount ? ` · ${scores.length} of ${seedCount} done` : ''}` : anyActive ? 'training…' : null,
      state: verdictSeen || scores.length === seedCount ? 'done' : anyActive || (running && scores.length > 0) ? 'active' : 'pending', tone: 'gray',
      what: 'The code trains once per seed, each run in its own sandbox branch with a different random start. Seed 0 runs first to catch crashes early; the others run in parallel.',
      why: 'One run can be lucky or unlucky. Several seeds show how much the result changes by chance. The verdict uses their mean.',
      facts: [
        ...(mean != null ? [{ label: 'Mean', value: `${fmtValue(metric, mean)} over ${scores.length} seed${scores.length > 1 ? 's' : ''}` }, { label: 'Range', value: `${fmtValue(metric, Math.min(...scores))} to ${fmtValue(metric, Math.max(...scores))}` }] : []),
        ...singles.map((n, i) => ({ label: `Seed ${i}`, value: n.sub ?? (n.state === 'pending' ? 'not started' : '') })),
      ],
      row: 5, col: 0, rowSize: 1,
    }
    seedNodes = [group]
  }
  const lastRow = 6

  // --- harness ---
  const ablation = of('ablation').pop()
  const harnessFacts: Fact[] = []
  if (model.attempts.length) harnessFacts.push({ label: 'Seeds scored', value: `${model.attempts.length}${model.seedsTotal ? ` of ${model.seedsTotal}` : ''}` })
  if (ablation) {
    const full = num(ablation.accuracy_full), abl = num(ablation.accuracy_ablated)
    const v = String(ablation.verdict ?? '')
    harnessFacts.push({ label: 'Graph check', value: `${v === 'Graph genutzt' ? 'passed: the code uses the graph' : v === 'Graph nicht genutzt' ? 'failed: the code ignores the graph' : v === 'unklar' ? 'inconclusive' : 'could not be measured'}${full != null && abl != null ? ` (${fmtValue('accuracy', full)} with edges, ${fmtValue('accuracy', abl)} without)` : ''}` })
  }
  if (faultMean) harnessFacts.push({ label: 'Fix after all seeds', value: `${faultMean}x: the mean was worse than a trivial baseline` })

  // --- verdict ---
  const verdictFacts: Fact[] = []
  if (verdictEv) {
    verdictFacts.push({ label: 'Mean over seeds', value: `${fmtValue(metric, num(verdictEv.mean))} ± ${fmtValue(metric, num(verdictEv.std))}` })
    if (model.target != null) verdictFacts.push({ label: 'Claimed', value: fmtValue(metric, model.target) })
    if (model.tolerance != null) verdictFacts.push({ label: 'Tolerance', value: `± ${fmtValue(metric, model.tolerance, 2)}` })
    verdictFacts.push({ label: 'Method fidelity', value: 'not checked' })
  }
  const vTone: Tone = verdictEv ? (verdictEv.urteil === 'reproduziert' ? 'green' : verdictEv.urteil === 'teilweise' ? 'amber' : verdictEv.urteil === 'nicht reproduziert' ? 'red' : 'gray') : 'gray'

  const sources: StepNode[] = []
  const addSrc = (n: Omit<StepNode, 'row' | 'col' | 'rowSize'>) => sources.push({ ...n, row: 1, col: 0, rowSize: 0 })
  addSrc({
    id: 'src_paper', kind: 'src_paper', title: 'Paper text', sub: srcSub(paperSrc, paperCall, !!target || has('plan')), state: st(!!paperSrc || !!paperCall || !!target || has('plan'), !paperSrc && active('claim')), tone: 'amber',
    what: 'Tavily fetches the text of the paper from arXiv.', why: 'The claimed number must come from the paper itself, not from the model’s memory.', facts: srcFacts(paperSrc, paperCall),
  })
  addSrc({
    id: 'src_readme', kind: 'src_readme', title: 'Code page (README)', sub: !readmeSrc && !readmeCall && (has('plan') || verdictSeen) ? 'no code page used' : srcSub(readmeSrc, readmeCall, has('setup_details')), state: st(!!readmeSrc || !!readmeCall, !readmeSrc && active('setup')), tone: 'amber',
    what: 'Tavily fetches the README of the authors’ code repository, if the paper links one.', why: 'Authors often state hyperparameters and data splits there. Reusing them is better than guessing.', facts: srcFacts(readmeSrc, readmeCall),
  })
  if (repoSearch) {
    addSrc({
      id: 'src_repo', kind: 'src_repo', title: 'Repository search', sub: repoSearch.url ? (repoSearch.live ? 'found, live search' : 'found') : 'nothing found', state: 'done', tone: 'amber',
      what: 'The paper names no code link, so Tavily searches the web for the authors’ repository.', why: 'A repository can contain details the paper leaves out. A hit only counts if its page names the paper.',
      facts: [{ label: 'Searched for', value: String(repoSearch.title ?? '') }, ...(repoSearch.url ? [{ label: 'Found', value: host(String(repoSearch.url)) }] : [])],
    })
  }
  if (errorSearches.length) {
    addSrc({
      id: 'src_error', kind: 'src_error', title: 'Error search', sub: `${errorSearches.length} web search${errorSearches.length > 1 ? 'es' : ''}`, state: 'done', tone: 'amber',
      what: 'When the code crashed, Tavily searched the web for the error message and handed the hits to the code model as hints.', why: 'A fix may already be written down somewhere. Only the error text leaves the system, never test data.',
      facts: errorSearches.flatMap((e, i) => [{ label: `Search ${i + 1}`, value: clip(String(e.query ?? ''), 140) }, ...((e.urls ?? []) as string[]).filter(Boolean).slice(0, 3).map((u) => ({ label: '  hit', value: host(String(u)) }))]),
    })
  }
  sources.forEach((n, i) => { n.col = i; n.rowSize = sources.length })

  const one = (n: Omit<StepNode, 'col' | 'rowSize'>): StepNode => ({ ...n, col: 0, rowSize: 1 })
  const nodes: StepNode[] = [
    one({
      id: 'paper', kind: 'paper', title: model.title ?? 'Paper', sub: runStart?.paper_id ? String(runStart.paper_id) : null, state: has('run_start') ? 'done' : 'pending', tone: 'gray',
      what: 'The paper whose claim is checked.', why: 'Nullius takes nobody’s word for it: it rebuilds the experiment instead of trusting the reported number.',
      facts: [...(runStart?.title ? [{ label: 'Title', value: String(runStart.title) }] : []), { label: 'Mode', value: 'Rebuild: the agent writes its own code' }], row: 0,
    }),
    ...sources,
    one({
      id: 'claim', kind: 'claim', title: 'Claim', sub: target ? `target ${fmtValue(metric, num(target.value), 2)}` : null, state: st(!!target || has('plan'), active('claim')), tone: 'green',
      what: 'Nemotron (Lightning) finds the reported number in the results table. We keep it only if the quote appears word for word in the text and the table column fits.',
      why: 'This is the number Nullius tries to reproduce. A misread would make everything after it meaningless.', facts: claimFacts, expand: target?.quote ? { kind: 'quote', quote: String(target.quote) } : undefined, row: 2,
    }),
    one({
      id: 'plan', kind: 'plan', title: 'Plan', sub: planEv ? 'written' : null, state: st(!!planEv && (has('code_written') || has('attempt')), !planEv || !has('code_written') ? active('plan') || active('setup') : false), tone: 'violet',
      what: 'Nemotron (Super) writes a short plan for the experiment from the paper and the setup details. Details without a matching quote are dropped.',
      why: 'A written plan keeps the code close to the paper’s method and makes the experiment easy to check.',
      facts: [...(setup ? [{ label: 'Setup details', value: `${setup.n ?? '?'} confirmed by quotes${num(setup.dropped) ? `, ${setup.dropped} dropped` : ''}` }] : []), ...planFacts],
      expand: planRow?.expand, row: 3,
    }),
    one({
      id: 'code', kind: 'code', title: 'Code', sub: codeVersions ? `version ${codeVersions}${codeVersions > 1 ? ' (rewritten)' : ''}` : null, state: st(has('code_written') && (has('attempt') || has('ablation')), active('code') || (has('code_written') && !has('attempt'))), tone: 'violet',
      what: 'Nemotron (Super) writes a training script. It must output predictions only.', why: 'The agent never reports a score itself and never sees the test labels, so it cannot tune its answer to the test.',
      facts: codeFacts, expand: lastCode?.expand, row: 4,
    }),
    ...seedNodes,
    one({
      id: 'harness', kind: 'harness', title: 'Independent harness', sub: model.attempts.length ? `${model.attempts.length} seed${model.attempts.length > 1 ? 's' : ''} scored` : null,
      state: st(verdictSeen, model.attempts.length > 0 || active('sandbox') || active('seeds')), tone: 'gray',
      what: 'Our own program loads the true test labels itself, checks the predictions and computes the score. For graph models it also runs the code again without graph edges.',
      why: 'The score never comes from the agent, so a wrong or lucky-looking number cannot be passed off as a result.', facts: harnessFacts, row: lastRow,
    }),
    one({
      id: 'verdict', kind: 'verdict', title: 'Verdict', sub: verdictEv ? (VERDICT_SENTENCE[String(verdictEv.urteil)] ?? String(verdictEv.urteil)) : null, state: verdictSeen ? 'done' : 'pending', tone: vTone,
      what: 'The harness compares the mean over all seeds with the claimed value, inside a tolerance fixed before the run.',
      why: 'The verdict says what happened with our setup. Whether our code follows the paper’s method is not checked.', facts: verdictFacts, row: lastRow + 1,
    }),
  ]

  // --- edges ---
  const edges: StepEdge[] = []
  const e = (source: string, target: string, extra: Partial<StepEdge> = {}) => edges.push({ id: `${source}>${target}`, source, target, ...extra })
  for (const s of sources) if (s.id !== 'src_error') e('paper', s.id)
  e('src_paper', 'claim')
  e('claim', 'plan')
  for (const s of sources) if (s.id === 'src_readme' || s.id === 'src_repo') e(s.id, 'plan')
  e('plan', 'code')
  if (errorSearches.length) e('src_error', 'code', { dashed: true, label: 'hints' })
  for (const s of seedNodes) { e('code', s.id); e(s.id, 'harness') }
  e('harness', 'verdict')
  if (seedLoop) e(seedNodes[0].id, 'code', { loop: 'left', dashed: true, label: `repairs: ${seedLoop}` })
  if (faultMean) e('harness', 'code', { loop: 'right', dashed: true, label: `fix after all seeds ×${faultMean}` })
  return { nodes, edges }
}

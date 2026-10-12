// Fixed benchmark figures for the two-mode view. Every number has a source in the results documents;
// nothing here is computed or estimated. Two modes, two separate measurements: they are never added up.
export const REPO_BLOB = 'https://github.com/Script-hpp/Nullius/blob/main'

export const SOURCES = {
  v4: 'benchmark/experiments/results/benchmark_v4.md',
  cg: 'benchmark/experiments/results/recall_cg_v1.md',
  readme: 'docs/evaluation_notes.md#our-own-small-bitter-lesson',
} as const

export type UnitId = 'kipf' | 'sgc' | 'dlinear' | 'nlinear' | 'shchurGcn' | 'shchurGat' | 'velGat' | 'dgi'
export type Unit = { id: UnitId; author: string; model: string; paperOnly: boolean; withCode: boolean }

// Match = majority verdict over the generations equals the reference verdict.
// paperOnly: benchmark v4, with Tavily (README table, column "Paper only"). withCode: recall cg v1 (results table, column "Treffer").
export const UNITS: Unit[] = [
  { id: 'kipf', author: 'Kipf', model: 'GCN', paperOnly: true, withCode: true },
  { id: 'sgc', author: 'Wu', model: 'SGC', paperOnly: false, withCode: true },
  { id: 'dlinear', author: 'Zeng', model: 'DLinear', paperOnly: false, withCode: true },
  { id: 'nlinear', author: 'Zeng', model: 'NLinear', paperOnly: false, withCode: false },
  { id: 'shchurGcn', author: 'Shchur', model: 'GCN', paperOnly: true, withCode: true },
  { id: 'shchurGat', author: 'Shchur', model: 'GAT', paperOnly: false, withCode: true },
  { id: 'velGat', author: 'Veličković', model: 'GAT', paperOnly: false, withCode: true },
  { id: 'dgi', author: 'Veličković', model: 'DGI', paperOnly: false, withCode: true },
]

export const HELD_OUT = { author: 'Li', model: 'RLinear', matched: 0, of: 1, measured: [0.487, 0.432], target: 0.366 } as const

export const count = (key: 'paperOnly' | 'withCode') => UNITS.filter((u) => u[key]).length

export const FIGURES = {
  placebo: { v2: [0, 48], v3: [0, 48], v4: [0, 48], cg: [0, 9] },
  reproducedAgree: { v4: [7, 7] },
  paperOnly: { matched: 2, of: 8, runs: 48 },
  withCode: { matched: 7, of: 8, countedRuns: 22 },
  singleGenerations: { matched: 15, of: 22 },
} as const

export const MODES = {
  paperOnly: { label: 'Paper only', detail: 'benchmark v4, with Tavily (stored results)', sub: 'majority of 3 generations, pre-registered', source: SOURCES.v4 },
  withCode: { label: "Paper and authors' code", detail: 'recall cg v1, Super in all roles', sub: 'majority of 2 to 3 generations', source: SOURCES.cg },
} as const

export const LIMITS: { title: string; text: string }[] = [
  { title: 'In-sample', text: 'The fixes were designed after seeing these eight units. Only the held-out paper (RLinear) was not used, and it missed in both generations (0 of 1).' },
  { title: 'No "recall improved" claim', text: 'The pre-registered reading rule required the held-out check as well, and it failed. 2 of 8 and 7 of 8 are never summed or netted into one gain.' },
  { title: 'Generation variance', text: 'Three of the seven matches (Shchur GAT, Veličković GAT, DGI) hang on a third generation; NLinear flipped from a match to a miss. Per single generation, 15 of 22 verdicts matched.' },
  { title: 'Different mode and models', text: "The authors'-code run used Nemotron 3 Super in every role (Ultra was removed from Token Factory on 2026-10-09), stored Tavily results and different run rules. It says nothing about Tavily's contribution." },
  { title: 'Method fidelity not checked', text: '"Reproduced" means the target value was reached according to the harness, not that the method was reproduced. The authors\' code is read as text and never executed.' },
]

export const BITTER_LESSON = 'Days of hand-built cleverness left the count at 2 of 8; the missing details were in the authors’ code, not in a cleverer prompt.'

// Wording for the start page and the judges page. Numbers are not repeated here: pages read FIGURES, HELD_OUT and UNITS.
export const REPO = 'https://github.com/Script-hpp/Nullius'
export const SUTTON_URL = 'http://www.incompleteideas.net/IncIdeas/BitterLesson.html'
export const README_TAVILY = `${REPO_BLOB}/docs/evaluation_notes.md#how-we-use-tavily`

// Scope limit stated in the README ("Limitations and roadmap") and CLAUDE.md; shown next to LIMITS on the start page.
export const SCOPE_LIMIT = { title: 'Narrow scope', text: 'Graph and time-series papers on CPU, on datasets with a ready-made task (Cora, ETTh1). No GPU training, no large models. A new dataset needs a task and a harness built once.' }

// Submission links appear in the footer once supplied; unpublished destinations are omitted.
export const SUBMISSION_LINKS: { label: string; href: string | null }[] = [
  { label: 'Demo video', href: null },
  { label: 'Devpost submission', href: null },
]

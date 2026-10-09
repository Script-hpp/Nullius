// Tabs of the run page. The tab lives in the URL (`#/run/<id>?tab=...`), so a link opens the same view. Older links used the
// previous tab names (`log`, `evidence`, `tree`, `files`, `graph`); they map to the nearest new tab.
export const RUN_TABS = ['overview', 'how', 'sources', 'code', 'limits'] as const
export type RunTab = (typeof RUN_TABS)[number]
export type HowView = 'graph' | 'log'

export const TAB_LABEL: Record<RunTab, string> = {
  overview: 'Overview',
  how: 'How we got there',
  sources: 'Sources & Tavily',
  code: 'Code & branches',
  limits: 'Limits',
}

const LEGACY: Record<string, { tab: RunTab; how?: HowView }> = {
  graph: { tab: 'how', how: 'graph' },
  log: { tab: 'how', how: 'log' },
  evidence: { tab: 'sources' },
  tree: { tab: 'code' },
  files: { tab: 'code' },
}

export function parseTab(q: string | null | undefined): { tab: RunTab; how: HowView } {
  const v = (q ?? '').toLowerCase()
  if ((RUN_TABS as readonly string[]).includes(v)) return { tab: v as RunTab, how: 'graph' }
  const l = LEGACY[v]
  return l ? { tab: l.tab, how: l.how ?? 'graph' } : { tab: 'overview', how: 'graph' }
}

// The hash for a tab: overview is the default and carries no query. Other query parts of the hash (play=1) are dropped on purpose.
export function tabHash(hash: string, tab: RunTab): string {
  const base = hash.replace(/^#/, '').split('?')[0]
  return `#${base}${tab === 'overview' ? '' : `?tab=${tab}`}`
}

import { AlertCircle, CheckCircle2, CircleDashed, Loader2, MinusCircle } from 'lucide-react'
import type { Event, Json } from '../../data'
import { fmtValue } from './phases'

// The numbered way from paper to verdict, with a status for each step. Built from the events and the dossier of a run (finished or live).
export type StepId = 'paper' | 'claim' | 'sources' | 'plan' | 'code' | 'sandbox' | 'check' | 'verdict' | 'community'
export type StepStatus = 'done' | 'running' | 'waiting' | 'skipped'
export type Step = { id: StepId; n: number; title: string; actor: string; what: string; status: StepStatus; data: string | null }

const DEF: { id: StepId; title: string; actor: string; what: string }[] = [
  { id: 'paper', title: 'Paper', actor: 'arXiv', what: 'The paper whose claim we check.' },
  { id: 'claim', title: 'Claim', actor: 'Tavily · Nemotron', what: 'The number the paper reports, with the sentence it comes from.' },
  { id: 'sources', title: 'Sources', actor: 'Tavily', what: 'Tavily reads the paper and its code page to collect the setup details.' },
  { id: 'plan', title: 'Plan', actor: 'Nemotron', what: 'A short plan for rebuilding the experiment.' },
  { id: 'code', title: 'Code', actor: 'Nemotron', what: 'The program that implements the plan. It never sees the true answers.' },
  { id: 'sandbox', title: 'Sandbox runs', actor: 'Nebius Sandboxes', what: 'The program runs in isolated sandboxes, one run per random seed.' },
  { id: 'check', title: 'Independent check', actor: 'Harness', what: 'Our harness, which the agent cannot change, scores the predictions.' },
  { id: 'verdict', title: 'Verdict', actor: 'Harness', what: 'The mean over all seeds is compared with the claim, inside a fixed tolerance.' },
  { id: 'community', title: 'What others found', actor: 'Tavily', what: 'Public discussion of the paper, listed next to the result. It never changes the verdict.' },
]

const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : x != null && x !== '' && Number.isFinite(Number(x)) ? Number(x) : null)

export function buildSteps(events: Event[], dossier: Json | null, o: { running: boolean; hasPlan?: boolean; metric?: string | null }): Step[] {
  const d = dossier ?? {}
  const has = (t: string) => events.some((e) => e.type === t)
  const claim = (d.claims ?? [])[0]
  const claimStart = events.find((e) => e.type === 'claim_start')
  const metric = o.metric ?? claim?.metric ?? (claimStart?.metric as string | undefined) ?? null
  const attempts: Json[] = (d.attempts ?? []).length ? d.attempts : events.filter((e) => e.type === 'attempt')
  const measured = attempts.filter((a) => num(a.measured) != null)
  const seedsDone = new Set(measured.map((a) => num(a.seed))).size
  const seedsTotal = num(events.find((e) => e.type === 'run_start')?.seeds) ?? new Set(attempts.map((a) => num(a.seed))).size
  const tavilyN = events.filter((e) => e.type === 'tavily_call' && num(e.role) !== 3 && num(e.role) !== 6).length
  const nSources = (d.sources ?? []).filter((s: Json) => s.kind !== 'community').length || (d.setup_details ?? []).length
  const verdict = (d.verdicts ?? [])[0] ?? events.filter((e) => e.type === 'verdict').pop()
  const target = num(claim?.target) ?? num(claimStart?.target)
  const communityRunning = has('tavily_call') && events.some((e) => e.type === 'tavily_call' && (num(e.role) === 3 || num(e.role) === 6)) && !has('community')
  const mean = num(verdict?.measured_mean ?? verdict?.mean)

  const done: Record<StepId, boolean> = {
    paper: has('run_start') || !!d.paper_id,
    claim: has('target_resolved') || has('claim_start') || !!claim,
    sources: tavilyN > 0 || has('source') || has('setup_details') || nSources > 0,
    plan: has('plan') || has('plan_text') || !!o.hasPlan,
    code: has('code_written') || has('code_text') || attempts.length > 0,
    sandbox: measured.length > 0,
    check: !!verdict,
    verdict: !!verdict,
    community: has('community') || d.community != null,
  }
  const data: Record<StepId, string | null> = {
    paper: null,
    claim: target != null ? `Target ${fmtValue(metric, target, 2)}` : null,
    sources: tavilyN ? `${tavilyN} Tavily ${tavilyN === 1 ? 'call' : 'calls'}` : nSources ? `${nSources} setup ${nSources === 1 ? 'detail' : 'details'} quoted` : null,
    plan: null,
    code: null,
    sandbox: seedsTotal ? `${seedsDone} of ${seedsTotal} seeds finished` : measured.length ? `${seedsDone} seeds finished` : null,
    check: measured.length ? `${measured.length} ${measured.length === 1 ? 'run' : 'runs'} scored` : null,
    verdict: mean != null ? `Mean ${fmtValue(metric, mean)}` : null,
    community: d.community ? `${(d.community.findings ?? []).length} ${(d.community.findings ?? []).length === 1 ? 'note' : 'notes'}` : null,
  }
  const order = DEF.map((s) => s.id)
  // on a running run the first unfinished step (before the community step) is running, later ones wait; a finished run marks the rest skipped
  const active = o.running ? order.filter((id) => id !== 'community').find((id) => !done[id]) ?? null : null
  return DEF.map((s, i) => {
    let status: StepStatus
    if (s.id === 'community') status = done.community ? 'done' : o.running ? (communityRunning ? 'running' : 'waiting') : 'skipped'
    else if (done[s.id]) status = 'done'
    else if (o.running) status = s.id === active ? 'running' : 'waiting'
    else status = 'skipped'
    return { ...s, n: i + 1, status, data: data[s.id] }
  })
}
export const STATUS_TEXT: Record<StepStatus, string> = { done: 'Done', running: 'Running', waiting: 'Waiting', skipped: 'Not part of this run' }

export function StatusMark({ s }: { s: StepStatus }) {
  const cls = 'h-4 w-4 shrink-0'
  if (s === 'done') return <CheckCircle2 className={`${cls} text-tone-green`} />
  if (s === 'running') return <Loader2 className={`${cls} animate-spin text-fg-2`} />
  if (s === 'skipped') return <MinusCircle className={`${cls} text-subtle`} />
  return <CircleDashed className={`${cls} text-subtle`} />
}

// Every step: number, title, one sentence on what happens, status, and (when known) one fact from this run. A click jumps to the step.
export function StepStrip({ steps, onStep }: { steps: Step[]; onStep: (id: StepId) => void }) {
  return (
    <section aria-label="The steps of a check" className="min-w-0">
      <div className="mb-6 flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">The nine steps of a check</h2>
        <span className="text-[12px] text-muted">Click a step to jump to its details.</span>
      </div>
      <ol className="m-0 grid list-none gap-0 border-t border-line p-0 md:grid-cols-3">
        {steps.map((s, i) => (
          <li key={s.id} className={`border-b border-line ${i % 3 ? 'md:border-l md:pl-8' : ''} ${i % 3 !== 2 ? 'md:pr-8' : ''}`}>
            <a href={`#step-${s.id}`} onClick={(e) => { e.preventDefault(); onStep(s.id) }}
              className={`block py-6 no-underline ${s.status === 'skipped' || s.status === 'waiting' ? 'opacity-70' : ''}`}>
              <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-2xl font-light text-fg-4">{String(s.n).padStart(2, '0')}</span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-fg-4"><StatusMark s={s.status} />{STATUS_TEXT[s.status]}</span>
              </div>
              <h3 className="font-grotesk m-0 mt-3 text-xl font-semibold tracking-[-0.03em] text-fg">{s.title}</h3>
              <div className="mt-0.5 text-[11px] uppercase tracking-wider text-muted">{s.actor}</div>
              <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4">{s.what}</p>
              {s.data && s.status !== 'skipped' && <p className="mb-0 mt-2 font-mono text-[12px] text-fg-3">{s.data}</p>}
            </a>
          </li>
        ))}
      </ol>
      {steps.some((s) => s.status === 'skipped') && (
        <p className="mb-0 mt-4 flex items-start gap-2 text-[12px] leading-relaxed text-muted"><AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          A step marked Not part of this run did not happen or was not recorded for it, for example runs from before that step existed.</p>
      )}
    </section>
  )
}

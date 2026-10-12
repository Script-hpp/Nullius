import type { Step, StepId } from './StepStrip'
import { StatusMark } from './StepStrip'

// The nine steps as nine quiet lines: number, status, title, one fact from this run. A click jumps to where the step is shown.
export function StepList({ steps, onStep }: { steps: Step[]; onStep: (id: StepId) => void }) {
  return (
    <ol className="m-0 list-none border-t border-line p-0" aria-label="The nine steps of a check">
      {steps.map((s) => (
        <li key={s.id} className="border-b border-line">
          <a href={`#step-${s.id}`} onClick={(e) => { e.preventDefault(); onStep(s.id) }} title={s.what}
            className={`my-1 grid grid-cols-[1.75rem_1.25rem_minmax(0,1fr)] items-center gap-x-3 rounded-lg px-3 py-2 no-underline transition-colors hover:bg-raised/40 sm:grid-cols-[1.75rem_1.25rem_minmax(0,1fr)_auto] ${s.status === 'skipped' || s.status === 'waiting' ? 'opacity-60' : ''}`}>
            <span className="font-mono text-[12px] text-muted">{String(s.n).padStart(2, '0')}</span>
            <StatusMark s={s.status} />
            <span className="min-w-0 text-[14px] text-fg-2">
              <span className="font-medium">{s.title}</span>
              <span className="ml-2 text-[11px] uppercase tracking-wider text-muted">{s.actor}</span>
            </span>
            {s.data && s.status !== 'skipped' && <span className="col-start-3 font-mono text-[12px] text-fg-4 sm:col-start-4">{s.data}</span>}
          </a>
        </li>
      ))}
    </ol>
  )
}

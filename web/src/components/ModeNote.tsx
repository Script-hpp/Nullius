import { FileCode2, FileText } from 'lucide-react'
import { runMode } from '../mode'
import type { Event, Json } from '../data'
import { Hint } from './Hint'

// One line under the claim: which way this check rebuilt the paper, and what that means. The label comes from the run itself
// (dossier or live `mode` event), so a fallback to the paper alone is never shown as an authors' code rebuild.
export function ModeNote({ dossier, events, className = '', compact = false }: { dossier?: Json; events: Event[]; className?: string; compact?: boolean }) {
  const mode = runMode(dossier, events)
  if (!mode) return null
  const Icon = mode.key === 'nachbau_autorencode' ? FileCode2 : FileText
  if (compact) {   // verdict card: label with an info icon, the explanation on hover or tap
    return (
      <span className={`inline-flex items-center gap-1.5 ${className}`} data-testid="run-mode">
        <Icon className="h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
        <span>{mode.label}{mode.planned ? ' (planned)' : ''}</span>
        <Hint text={mode.explain + (mode.planned ? ' The mode is confirmed once the repository has been read.' : '')} />
      </span>
    )
  }
  return (
    <p className={`m-0 flex max-w-3xl items-start gap-3 text-[13px] leading-relaxed text-fg-4 ${className}`} data-testid="run-mode">
      <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted" aria-hidden />
      <span>
        <b className="font-semibold text-fg-3">{mode.label}{mode.planned ? ' (planned)' : ''}.</b> {mode.explain}
        {mode.planned && ' The mode is confirmed once the repository has been read; without usable authors’ code the check runs from the paper alone.'}
      </span>
    </p>
  )
}
